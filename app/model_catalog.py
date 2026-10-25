"""Approved checkpoints, pinned revisions, and independent local caches."""
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class ModelSpec:
    variant: str
    model_id: str
    revision: str
    cache_name: str
    family: str
    longest_edge: int

    @property
    def directory(self) -> Path:
        return APP_DIR / "models" / self.cache_name

MODELS = {
    "0.8B": ModelSpec("0.8B", "lightonai/LightOnOCR-3-0.8B", "36d8636925327c671bd8e479c517a004f7566bd9", "lightonocr-0.8b", "qwen3_5", 2048),
    "1B": ModelSpec("1B", "lightonai/LightOnOCR-3-1B", "23d544085c886bbdace647bcfab91203299417e2", "lightonocr", "lighton_ocr", 1540),
    "4B": ModelSpec("4B", "lightonai/LightOnOCR-3-4B", "b71010f095dc3735de7ee6b9969bff8f8f50b39a", "lightonocr-4b", "qwen3_5", 2048),
}

def get_model(variant: str) -> ModelSpec:
    try:
        return MODELS[variant]
    except (KeyError, TypeError) as exc:
        raise ValueError("Choose Start 0.8B, Start 1B, or Start 4B in Pinokio.") from exc

def download_model(spec: ModelSpec) -> None:
    # Called only after an explicit selection; the existing 1B directory is reused.
    from huggingface_hub import snapshot_download
    print(f"Preparing {spec.model_id} at {spec.revision}; cached files are reused.", flush=True)
    snapshot_download(spec.model_id, revision=spec.revision, local_dir=str(spec.directory),
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "README.md"],
                      max_workers=2)
