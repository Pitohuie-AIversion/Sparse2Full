import torch
import psutil
import platform
import os
import sys

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def print_system_info():
    print("="*40, "System Information", "="*40)
    try:
        uname = platform.uname()
        print(f"System: {uname.system}")
        print(f"Node Name: {uname.node}")
        print(f"Release: {uname.release}")
        print(f"Machine: {uname.machine}")
    except:
        pass
    
    print("\n" + "="*40, "CPU Info", "="*40)
    try:
        print(f"Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"Total cores: {psutil.cpu_count(logical=True)}")
        cpufreq = psutil.cpu_freq()
        if cpufreq:
            print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
            print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    except:
        pass
    
    print("\n" + "="*40, "Memory Information", "="*40)
    try:
        svmem = psutil.virtual_memory()
        print(f"Total: {get_size(svmem.total)}")
        print(f"Available: {get_size(svmem.available)}")
        print(f"Used: {get_size(svmem.used)} ({svmem.percent}%)")
    except:
        pass
    
    print("\n" + "="*40, "GPU Information", "="*40)
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"  Total Memory: {get_size(props.total_memory)}")
                print(f"  Multi Processor Count: {props.multi_processor_count}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
            except:
                pass
    else:
        print("CUDA Available: No")

if __name__ == "__main__":
    print_system_info()
