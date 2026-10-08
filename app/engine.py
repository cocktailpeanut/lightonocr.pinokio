"""Local Transformers inference for the pinned LightOnOCR-3-1B checkpoint."""
from __future__ import annotations

import logging
import os
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

MODEL_ID = "lightonai/LightOnOCR-3-1B"
MODEL_REVISION = "b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "lightonocr"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LONGEST_EDGE = 1540
Image.MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("lightonocr")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class Engine:
    def __init__(self) -> None:
        import torch
        import transformers
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

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
        if not (MODEL_DIR / "config.json").is_file():
            raise RuntimeError("Model is not installed. Run the launcher Install action first.")
        logger.info("Loading %s on %s (%s) from local files", MODEL_ID, self.device, dtype_name)
        self.processor = LightOnOcrProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
        # This pinned checkpoint retains the legacy nested CausalLM key prefix.
        # Transformers' native LightOnOcr class embeds the base language model.
        self.model, loading = LightOnOcrForConditionalGeneration.from_pretrained(
            str(MODEL_DIR), dtype=self.dtype, local_files_only=True,
            key_mapping={r"^language_model\.model\.": "model.language_model."},
            output_loading_info=True,
        )
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
                "dtype": self.dtype_name, "model": MODEL_ID, "revision": MODEL_REVISION,
                "versions": self.versions, "busy": self.inference_lock.locked(),
                "weight_loading": {"strict": True, **self.loading_info},
                "limits": {"upload_mb": 25, "longest_edge": LONGEST_EDGE,
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


def normalize_image(image: Image.Image) -> Image.Image:
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
    result.thumbnail((LONGEST_EDGE, LONGEST_EDGE), Image.Resampling.LANCZOS)
    return result
