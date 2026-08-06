from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src.features.semantic_analysis.pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline


logger = logging.getLogger("fabricvision.semantic_analysis.runner")


def load_config(config_path: Optional[str] = None) -> SemanticAnalysisConfig:
    """Load Semantic Analysis configuration from the default config path."""
    workspace_root = Path(__file__).resolve().parents[2]
    config = SemanticAnalysisConfig()
    config.config_dir = str(workspace_root / "configs")
    config.output_root = str(workspace_root / "curated_dataset")
    config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    if config_path:
        config.config_path = str(Path(config_path).expanduser().resolve())
    return config


def main() -> None:
    """Run Semantic Analysis for a single processed garment image from the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 2:
        logger.error("No input image provided. Usage: python -m src.features.semantic_analysis.run_semantic_analysis <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser().resolve()
    if not image_path.exists():
        logger.error("Input image does not exist: %s", image_path)
        sys.exit(1)

    if not image_path.is_file():
        logger.error("Input path is not a file: %s", image_path)
        sys.exit(1)

    try:
        logger.info("Semantic Analysis Started")
        logger.info("Loading configuration...")
        config = load_config()

        logger.info("Loading Semantic Analysis pipeline...")
        pipeline = SemanticAnalysisPipeline(config=config)

        logger.info("Analyzing image: %s", image_path)
        start_time = time.perf_counter()
        result = pipeline.run(str(image_path))
        elapsed_seconds = time.perf_counter() - start_time

        if result.get("status") == "completed":
            logger.info("Generating metadata...")
            logger.info("Validating metadata...")
            logger.info("Organizing garment...")
            logger.info("Semantic Analysis Completed")
            print("\nSemantic Analysis Completed")
            print(f"Input Image: {image_path}")
            print(f"Metadata: {result.get('metadata_path')}")
            print(f"Curated Image: {result.get('organized_image_path')}")
            print(f"Execution Time: {elapsed_seconds:.2f}s")
        else:
            logger.error("Semantic Analysis failed: %s", result.get("issues", []))
            print("\nSemantic Analysis Failed")
            print(f"Input Image: {image_path}")
            for issue in result.get("issues", []):
                print(f"- {issue.get('field')}: {issue.get('message')}")
            sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Input image error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - user-facing guard
        logger.error("Semantic Analysis execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
