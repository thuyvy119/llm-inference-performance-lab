from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
from client.request import InferenceClient
from configs.loader import load_yaml
from metrics.gpu import GPUSampler
from metrics.latency import summarize

CONFIG_PATH = Path("configs/experiments/exp2_concurrency.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")
OUTPUT_PATH = Path("results/raw/exp2_concurrency.json")
GPU_SAMPLING_INTERVAL_SECONDS = 0.05

