from __future__ import annotations

from pathlib import Path

import yaml

from src.features.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline


def load_config(config_path: str | Path | None = None) -> PreprocessingConfig:
    if config_path is None:
        p = Path("configs/preprocessing/preprocessing_config.yaml")
        if not p.exists():
            p = Path("configs/preprocessing_config.yaml")
        config_path = p
    else:
        config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    return PreprocessingConfig(**raw_config)


def main() -> None:
    pipeline = PreprocessingPipeline(config=load_config())
    stats = pipeline.process_dataset()
    print("Preprocessing completed")
    print(f"Processed: {stats['processed_count']}")
    print(f"Failed: {stats['failed_count']}")
    print(f"Output dir: {stats['output_dir']}")


if __name__ == "__main__":
    main()
