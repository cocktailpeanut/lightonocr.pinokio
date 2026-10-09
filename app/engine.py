"""Local Transformers inference for an explicitly selected LightOnOCR-3 checkpoint."""
from __future__ import annotations

import logging
import os
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from PIL import Image, ImageOps
from model_catalog import get_model

MODEL_ID = get_model("1B").model_id
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LONGEST_EDGE = get_model("1B").longest_edge
Image.MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("lightonocr")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class Engine:
    def __init__(self, variant: str) -> None:
        import torch
        import transformers

        self.spec = get_model(variant)
        self.model_id = self.spec.model_id
        self.revision = self.spec.revision
        self.longest_edge = self.spec.longest_edge
        self.torch = torch
        self.inference_lock = threading.Lock()
        self.last_prompt_tokens = 0
        requested = os.getenv("LIGHTONOCR_DEVICE", "auto").strip().lower()
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise RuntimeError("LIGHTONOCR_DEVICE must be auto, cpu, cuda, or mps.")
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        self.device = ("cuda" if cuda else "mps" if mps else "cpu") if requested == "auto" else requested
        if self.device == "cuda" and not cuda:
            raise RuntimeError("CUDA was requested but is unavailable. Use LIGHTONOCR_DEVICE=cpu or auto.")
        if self.device == "mps" and not mps:
            raise RuntimeError("Apple MPS was requested but is unavailable. Use LIGHTONOCR_DEVICE=cpu or auto.")
        dtype_name = os.getenv("LIGHTONOCR_DTYPE", "auto").strip().lower()
        if dtype_name == "auto":
            if self.device == "mps":
                dtype_name = "float32"
            elif self.device == "cuda":
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype_name = "bfloat16"
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise RuntimeError("LIGHTONOCR_DTYPE must be auto, float32, float16, or bfloat16.")
        if self.device == "mps" and dtype_name != "float32":
            raise RuntimeError("MPS requires LIGHTONOCR_DTYPE=float32 for this model.")
        if self.device == "cpu" and dtype_name == "float16":
            raise RuntimeError("Use bfloat16 or float32 on CPU.")
        self.dtype_name = dtype_name
        self.dtype = getattr(torch, dtype_name)
        try:
            threads = int(os.getenv("LIGHTONOCR_CPU_THREADS", "4"))
        except ValueError as exc:
            raise RuntimeError("LIGHTONOCR_CPU_THREADS must be an integer from 1 to 32.") from exc
        if not 1 <= threads <= 32:
            raise RuntimeError("LIGHTONOCR_CPU_THREADS must be an integer from 1 to 32.")
        torch.set_num_threads(min(threads, os.cpu_count() or 1))
        model_dir = self.spec.directory
        if not (model_dir / "config.json").is_file():
            raise RuntimeError("The selected model is not cached. Start it through Pinokio to download it.")
        logger.info("Loading %s on %s (%s) from local files", self.model_id, self.device, dtype_name)
        options = {"dtype": self.dtype, "local_files_only": True, "output_loading_info": True}
        if self.spec.family == "lighton_ocr":
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
            self.processor = LightOnOcrProcessor.from_pretrained(str(model_dir), local_files_only=True)
            # Only the pinned 1B checkpoint has this legacy nested key prefix.
            options["key_mapping"] = {r"^language_model\.model\.": "model.language_model."}
            loader = LightOnOcrForConditionalGeneration
        else:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            loader = Qwen3_5ForConditionalGeneration
        self.model, loading = loader.from_pretrained(str(model_dir), **options)
        # Transformers may return sets; the original viewer uses stdlib json.dumps.
        self.loading_info = {
            key: list(values) if isinstance(values, (set, tuple)) else values
            for key, values in loading.items()
        }
        failures = {key: values for key, values in loading.items() if values}
        if failures:
            raise RuntimeError(f"Checkpoint weights did not load exactly; refusing to start: {failures}")
        logger.info("All checkpoint weights loaded; no missing, unexpected, or mismatched keys")
        self.model = self.model.to(self.device).eval()
        self.versions = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "fastapi": package_version("fastapi"),
            "pillow": package_version("Pillow"), "openai": package_version("openai"),
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ready", "backend": "transformers", "device": self.device,
                "dtype": self.dtype_name, "model": self.model_id, "revision": self.revision,
                "variant": self.spec.variant, "architecture": self.spec.family,
                "versions": self.versions, "busy": self.inference_lock.locked(),
                "weight_loading": {"strict": True, **self.loading_info},
                "limits": {"upload_mb": 25, "longest_edge": self.longest_edge,
                           "min_tokens": 128, "max_tokens": 4096}}

    def transcribe(self, image: Image.Image, mode: str, max_new_tokens: int,
                   temperature: float = 0.2, top_p: float = 1.0) -> tuple[str, int, bool]:
        # The adapter holds inference_lock for this call and usage collection.
        if mode not in {"plain", "grounding"}:
            raise ValueError("Mode must be plain or grounding.")
        content: list[dict[str, Any]] = [{"type": "image", "image": image}]
        if mode == "grounding":
            content.append({"type": "text", "text": "grounding"})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        inputs = {key: value.to(device=self.device, dtype=self.dtype)
                  if value.is_floating_point() else value.to(self.device)
                  for key, value in inputs.items()}
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])
        generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation.update(temperature=temperature, top_p=top_p)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        token_count = int(generated.shape[0])
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        ended = token_count > 0 and int(generated[-1]) in eos_ids
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, token_count, token_count >= max_new_tokens and not ended


def normalize_image(image: Image.Image, longest_edge: int = LONGEST_EDGE) -> Image.Image:
    transposed = ImageOps.exif_transpose(image)
    try:
        if transposed.mode in {"RGBA", "LA"} or "transparency" in transposed.info:
            rgba = transposed.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            try:
                result = Image.alpha_composite(background, rgba).convert("RGB")
            finally:
                rgba.close()
                background.close()
        else:
            result = transposed.convert("RGB")
    finally:
        transposed.close()
    result.thumbnail((longest_edge, longest_edge), Image.Resampling.LANCZOS)
    return result
