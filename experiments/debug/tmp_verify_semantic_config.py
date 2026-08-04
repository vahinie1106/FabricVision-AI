from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.semantic_analysis.pipeline import SemanticAnalysisPipeline, SemanticAnalysisConfig

config = SemanticAnalysisConfig(config_dir='configs', output_root='curated_dataset', device='cpu')
pipeline = SemanticAnalysisPipeline(config=config, inference_engine=None, model_loader=None)
print('model_path', pipeline.config.model_path)
print('model_exists', Path(pipeline.config.model_path).exists())
print('output_root', pipeline.config.output_root)
