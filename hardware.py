import os
import platform
import subprocess
import logging
import torch
import shutil

logger = logging.getLogger(__name__)

def get_cpu_model() -> str:
    """Retrieves the CPU Model Name robustly across Windows and Linux."""
    sys_type = platform.system()
    try:
        if sys_type == "Windows":
            cmd = "wmic cpu get name"
            output = subprocess.check_output(cmd, shell=True).decode(encoding="utf-8", errors="ignore").strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]
        elif sys_type == "Linux":
            # Kaggle / Colab container check
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            # Fallback
            cmd = "lscpu | grep 'Model name'"
            output = subprocess.check_output(cmd, shell=True).decode(encoding="utf-8", errors="ignore").strip()
            if ":" in output:
                return output.split(":")[1].strip()
    except Exception as e:
        logger.debug(f"Failed to fetch detailed CPU model: {e}")
        
    return platform.processor() or "Unknown Processor"


def get_system_ram_gb() -> float:
    """Retrieves Total System RAM as a float in GB."""
    sys_type = platform.system()
    try:
        if sys_type == "Windows":
            cmd = "wmic ComputerSystem get TotalPhysicalMemory"
            output = subprocess.check_output(cmd, shell=True).decode(encoding="utf-8", errors="ignore").strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return int(lines[1]) / (1024 ** 3)
        elif sys_type == "Linux":
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb_ram = int(line.split()[1])
                            return kb_ram / (1024 ** 2)
    except Exception:
        pass
    return 16.0  # Safe fallback if detection fails


def get_system_ram() -> str:
    """Retrieves Total System RAM installed robustly across Windows and Linux."""
    ram_gb = get_system_ram_gb()
    return f"{round(ram_gb, 2)} GB"


def get_gpu_vram_gb() -> float:
    """Retrieves Total GPU VRAM as a float in GB."""
    if torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(0)
            return properties.total_memory / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def get_available_disk_space() -> str:
    """Retrieves available disk space on the current drive."""
    try:
        _, _, free = shutil.disk_usage(".")
        return f"{round(free / (1024 ** 3), 2)} GB"
    except Exception:
        return "Unknown"


def get_hardware_diagnostics() -> dict:
    """
    Performs complete zero-dependency hardware profiling diagnostics.
    
    Returns:
        dict: Complete dictionary containing system metadata.
    """
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_vram = None
    gpu_vram_gb = get_gpu_vram_gb()
    ram_gb = get_system_ram_gb()
    
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = f"{round(gpu_vram_gb, 2)} GB"
        except Exception as e:
            logger.debug(f"Failed to fetch detailed GPU diagnostics: {e}")
            gpu_name = "NVIDIA CUDA GPU (Properties Unreadable)"
            gpu_vram = "Unknown VRAM Size"
            
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu": get_cpu_model(),
        "cpu_core_count": os.cpu_count() or 1,
        "ram": get_system_ram(),
        "ram_gb": ram_gb,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name if gpu_name else "N/A",
        "gpu_vram": gpu_vram if gpu_vram else "N/A",
        "gpu_vram_gb": gpu_vram_gb,
        "available_disk_space": get_available_disk_space()
    }


def print_hardware_summary():
    """Prints a beautiful scientific system profile diagnostic header."""
    summary = get_hardware_diagnostics()
    print("\n================ SYSTEM HARDWARE ENVIRONMENT DIAGNOSTICS ================")
    print(f" Operating System  : {summary['os']} (Version {summary['os_release']})")
    print(f" CPU Processor     : {summary['cpu']}")
    print(f" Physical RAM      : {summary['ram']}")
    print(f" CUDA Acceleration : {'ENABLED' if summary['cuda_available'] else 'DISABLED'}")
    if summary['cuda_available']:
        print(f" GPU Model Name    : {summary['gpu_name']}")
        print(f" GPU Video RAM     : {summary['gpu_vram']}")
    print("=========================================================================\n")
