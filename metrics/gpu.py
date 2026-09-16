from __future__ import annotations
import csv
import subprocess
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

class GPUSampler:
    def __init__(self, output_path: str | Path, interval_seconds: float = 0.05):
        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop_event = Event()
        self._thread: Thread | None = None

    # collect one sample from every visible GPU
    def sample_once(self) -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu="
            "index,"
            "name,"
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "temperature.gpu,"
            "power.draw",
            "--format=csv,noheader,nounits",
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        timestamp = time.perf_counter()
        samples = []

        for line in result.stdout.strip().splitlines():
            values = [value.strip() for value in line.split(",")]

            if len(values) != 7:
                continue

            (gpu_index, gpu_name, gpu_utilization, memory_used, memory_total, temperature, power) = values

            samples.append(
                {
                    "timestamp": timestamp,
                    "gpu_index": int(gpu_index),
                    "gpu_name": gpu_name,
                    "gpu_utilization_percent": float(
                        gpu_utilization
                    ),
                    "memory_used_mb": float(memory_used),
                    "memory_total_mb": float(memory_total),
                    "temperature_c": float(temperature),
                    "power_w": float(power),
                }
            )

        return samples

    def _run(self) -> None:
        """Continuously collect GPU samples."""
        while not self._stop_event.is_set():
            try:
                samples = self.sample_once()
                self.samples.extend(samples)
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                ValueError,
            ):
                # GPU monitoring failure should not crash the inference benchmark
                pass

            self._stop_event.wait(self.interval_seconds)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("GPU sampler is already running.")

        self.samples = []
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self._write_csv()

        return self.samples

    def _write_csv(self):
        if not self.samples:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "gpu_index",
            "gpu_name",
            "gpu_utilization_percent",
            "memory_used_mb",
            "memory_total_mb",
            "temperature_c",
            "power_w",
        ]

        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)