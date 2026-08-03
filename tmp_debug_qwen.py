from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.semantic_analysis.model.qwen_model import QwenModelLoader
from src.semantic_analysis.prompting.prompt_builder import PromptBuilder
from src.semantic_analysis.inference.qwen_inference import QwenInferenceEngine

image_path = Path('data/processed/garments/shirts/shirt_001.jpg')
prompt = PromptBuilder('configs').build(image_path)
loader = QwenModelLoader('models/Qwen2.5-VL-3B-Instruct', 'cpu')
loader.load()
engine = QwenInferenceEngine(loader, 'cpu')
response = engine.run(image_path, prompt)
print('RAW_RESPONSE_START', flush=True)
print(response, flush=True)
print('RAW_RESPONSE_END', flush=True)
