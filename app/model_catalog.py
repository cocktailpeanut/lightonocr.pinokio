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
    "0.8B": ModelSpec("0.8B", "lightonai/LightOnOCR-3-0.8B", "4a953edfc77f0e435532c503dd69dd74663d44c3", "lightonocr-0.8b", "qwen3_5", 2048),
    "1B": ModelSpec("1B", "lightonai/LightOnOCR-3-1B", "b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86", "lightonocr", "lighton_ocr", 1540),
    "4B": ModelSpec("4B", "lightonai/LightOnOCR-3-4B", "a06e5c5459551c9d1696468aece70ffbcf01ae62", "lightonocr-4b", "qwen3_5", 2048),
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
