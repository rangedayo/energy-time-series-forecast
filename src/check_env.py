import sys
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
except ImportError as e:
    sys.exit(f"ERROR: torch 임포트 실패 → {e}")

try:
    import xgboost
    print(f"XGBoost: {xgboost.__version__}")
except ImportError as e:
    sys.exit(f"ERROR: xgboost 임포트 실패 → {e}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except ImportError as e:
    sys.exit(f"ERROR: pandas 임포트 실패 → {e}")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    sys.exit(f"ERROR: numpy 임포트 실패 → {e}")

try:
    import statsmodels
    print(f"statsmodels: {statsmodels.__version__}")
except ImportError as e:
    sys.exit(f"ERROR: statsmodels 임포트 실패 → {e}")

try:
    import onnx
    import onnxruntime
    print(f"ONNX: {onnx.__version__}  /  ONNXRuntime: {onnxruntime.__version__}")
except ImportError as e:
    sys.exit(f"ERROR: onnx/onnxruntime 임포트 실패 → {e}")

print("\n✅ 환경 검증 통과")
