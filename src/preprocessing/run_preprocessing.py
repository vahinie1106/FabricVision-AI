from __future__ import annotations

from pathlib import Path

import yaml

from src.preprocessing.preprocessing_pipeline import PreprocessingConfig, PreprocessingPipeline


def load_config(config_path: str | Path | None = None) -> PreprocessingConfig:
    config_path = Path(config_path or "configs/preprocessing_config.yaml")
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
