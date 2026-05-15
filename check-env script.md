# 환경 검증 스크립트

```python
# 환경 검증 스크립트
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")

import pytorch_lightning as pl
print(f"PyTorch Lightning: {pl.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

from datasets import load_dataset
print("Hugging Face Datasets: OK")

print("\n✅ 모든 라이브러리가 정상적으로 설치되었습니다!")
```