"""Download the pretrained backbones used by the five benchmark repositories."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path('/root/autodl-tmp/baseline/models')
ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf')

MODELS = [
    ('google-bert/bert-base-uncased', 'bert-base-uncased'),
    ('openai-community/gpt2', 'gpt2'),
    ('DecisionIntelligence/Aurora', 'aurora'),
]

for model_id, name in MODELS:
    print(f'Downloading {model_id}', flush=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(ROOT / name),
        allow_patterns=['*.json', '*.txt', '*.safetensors', '*.py', '*.model',
                        'merges.txt', 'vocab.json'],
        max_workers=4,
    )
    print(f'Downloaded {name}', flush=True)
