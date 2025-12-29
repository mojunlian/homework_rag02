import torch
import onnxruntime as ort
import subprocess

def check_gpu():
    print("="*30)
    print("GPU Availability Check")
    print("="*30)

    # 1. PyTorch Check
    print("\n[PyTorch]")
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available for PyTorch. It will use CPU.")

    # 2. ONNX Runtime Check
    print("\n[ONNX Runtime]")
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("ONNX Runtime can use CUDA.")
    else:
        print("ONNX Runtime cannot use CUDA (CUDAExecutionProvider not found).")

    # 3. System Check (nvidia-smi)
    print("\n[System - nvidia-smi]")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("nvidia-smi found. GPU is recognized by the system.")
            # Print only the first few lines of nvidia-smi for brevity
            print("\n".join(result.stdout.split('\n')[:10]))
        else:
            print("nvidia-smi failed. GPU driver might not be installed or no NVIDIA GPU found.")
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")

if __name__ == "__main__":
    check_gpu()
