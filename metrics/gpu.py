import subprocess
import time

def get_gpu_metrics():
    command = [
        "nvidia-smi",
        "--query-gpu="
        "name,"
        "utilization.gpu,"
        "memory.used,"
        "memory.total,"
        "temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    line = result.stdout.strip()
    values = [val.strip() for val in line.split(",")]
    (name, gpu_utilization, memory_used, memory_total, temperature) = values
    return {
        "gpu_name": name,
        "gpu_utilization_percent": float(gpu_utilization),
        "gpu_memory_used_mb": float(memory_used),
        "gpu_memory_total_mb": float(memory_total),
        "temperature_c": float(temperature),
    }

def print_gpu_metrics(metrics):
    metrics = get_gpu_metrics()
    print("\n=== GPU Metrics ===")
    print(f"GPU name: {metrics['gpu_name']}")
    print(f"GPU Utilization: {metrics['gpu_utilization_percent']}%")
    print(
        f"Memory:       "
        f"{metrics['memory_used_mb']:.0f} / "
        f"{metrics['memory_total_mb']:.0f} MB"
    )
    print(f"GPU Temperature: {metrics['temperature_c']} °C")
    
if __name__ == "__main__":
    print_gpu_metrics()
