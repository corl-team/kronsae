"""Write model weights and configuration in the Hugging Face checkpoint format."""

from dataclasses import asdict
import json
from pathlib import Path

from .config import TrainingConfig


def save_checkpoint_files(sae, cfg: TrainingConfig, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # torch.compile wraps the SAE and prefixes its state_dict keys with _orig_mod.
    model = getattr(sae, "_orig_mod", sae)
    model.save_pretrained(directory, safe_serialization=True)

    training_config = asdict(cfg)
    for name in ("dtype", "sae_dtype"):
        training_config[name] = str(training_config[name]).removeprefix("torch.")
    with (directory / "training_config.json").open("w") as file:
        json.dump(training_config, file, indent=2, default=str)
