from __future__ import annotations
import csv
import subprocess
import pytest
from metrics.gpu import GPUSampler


def test_sample_once_parses_gpu_output(monkeypatch):
    fake_output = (
        "0, NVIDIA GeForce RTX 3090, 75, 12000, 24576, 65, 250.5\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=fake_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUSampler("results/test_gpu.csv")
    samples = sampler.sample_once()
    assert len(samples) == 1
    sample = samples[0]

    assert sample["gpu_index"] == 0
    assert sample["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert sample["gpu_utilization_percent"] == 75.0
    assert sample["memory_used_mb"] == 12000.0
    assert sample["memory_total_mb"] == 24576.0
    assert sample["temperature_c"] == 65.0
    assert sample["power_w"] == 250.5
    assert isinstance(sample["timestamp"], float)


def test_sample_once_parses_multiple_gpus(monkeypatch):
    fake_output = (
        "0, NVIDIA GeForce RTX 3090, 75, 12000, 24576, 65, 250.5\n"
        "1, NVIDIA GeForce RTX 3090, 40, 8000, 24576, 60, 180.0\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=fake_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUSampler("results/test_gpu.csv")
    samples = sampler.sample_once()
    assert len(samples) == 2
    assert samples[0]["gpu_index"] == 0
    assert samples[1]["gpu_index"] == 1


def test_sample_once_ignores_malformed_lines(monkeypatch):
    fake_output = (
        "0, NVIDIA GeForce RTX 3090, 75, 12000, 24576, 65, 250.5\n"
        "malformed,line\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=fake_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUSampler("results/test_gpu.csv")
    samples = sampler.sample_once()
    assert len(samples) == 1
    assert samples[0]["gpu_index"] == 0


def test_gpu_monitoring_failure_does_not_crash(monkeypatch):
    def fake_sample_once():
        raise FileNotFoundError("nvidia-smi not found")

    sampler = GPUSampler(
        "results/test_gpu.csv",
        interval_seconds=0.01,
    )

    monkeypatch.setattr(sampler, "sample_once", fake_sample_once)
    sampler.start()
    sampler.stop()

    assert sampler.samples == []


def test_stop_writes_csv(tmp_path, monkeypatch):
    fake_output = (
        "0, NVIDIA GeForce RTX 3090, 75, 12000, 24576, 65, 250.5\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=fake_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    output_path = tmp_path / "gpu_metrics.csv"
    sampler = GPUSampler(
        output_path=output_path,
        interval_seconds=0.01,
    )

    sampler.start()
    sampler.stop()
    assert output_path.exists()
    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= 1
    assert rows[0]["gpu_index"] == "0"
    assert rows[0]["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert rows[0]["gpu_utilization_percent"] == "75.0"
    assert rows[0]["memory_used_mb"] == "12000.0"
    assert rows[0]["memory_total_mb"] == "24576.0"
    assert rows[0]["temperature_c"] == "65.0"
    assert rows[0]["power_w"] == "250.5"

def test_stop_writes_empty_csv_when_gpu_unavailable(tmp_path, monkeypatch):
    def fake_sample_once():
        raise FileNotFoundError("nvidia-smi not found")

    output_path = tmp_path / "gpu_metrics.csv"

    sampler = GPUSampler(
        output_path=output_path,
        interval_seconds=0.01,
    )

    monkeypatch.setattr(sampler, "sample_once", fake_sample_once)

    sampler.start()
    sampler.stop()

    assert output_path.exists()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert rows == []

def test_sample_once_propagates_nvidia_smi_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GPUSampler("results/test_gpu.csv")
    with pytest.raises(subprocess.CalledProcessError):
        sampler.sample_once()