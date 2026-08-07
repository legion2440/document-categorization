#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model_optimization import export_quantized_tflite

if __name__ == "__main__":
    print(export_quantized_tflite(ROOT / "models/checkpoints"))
