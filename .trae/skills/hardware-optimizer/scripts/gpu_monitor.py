import time
import subprocess
import sys
from datetime import datetime

def get_gpu_utilization():
    try:
        # Check if nvidia-smi is available and working
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"], 
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        gpus = []
        for line in lines:
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) >= 3:
                util = int(parts[0].strip())
                mem_used = int(parts[1].strip())
                mem_total = int(parts[2].strip())
                gpus.append({'util': util, 'mem_used': mem_used, 'mem_total': mem_total})
        return gpus
    except Exception as e:
        return []

def monitor(duration=60, interval=1):
    print(f"Monitoring GPU usage for {duration} seconds (Ctrl+C to stop)...")
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            gpus = get_gpu_utilization()
            timestamp = datetime.now().strftime("%H:%M:%S")
            if not gpus:
                print(f"[{timestamp}] No GPU detected or nvidia-smi failed.", end='\r')
            else:
                status_parts = [f"[{timestamp}]"]
                for i, gpu in enumerate(gpus):
                    status_parts.append(f"GPU{i}: {gpu['util']}% Util, {gpu['mem_used']}/{gpu['mem_total']} MB")
                print(" | ".join(status_parts), end='\r')
            
            sys.stdout.flush()
            time.sleep(interval)
        print("\nMonitoring complete.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    try:
        duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        monitor(duration)
    except ValueError:
        print("Usage: python gpu_monitor.py [duration_seconds]")
