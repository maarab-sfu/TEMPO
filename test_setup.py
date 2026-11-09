import torch
import cv2
import transformers
import faiss
import sentence_transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("OpenCV:", cv2.__version__)
print("Transformers:", transformers.__version__)
print("FAISS:", faiss.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)

# Verify actual GPU computation
x = torch.randn(1000, 1000, device="cuda")
y = x @ x

print("GPU computation successful:", y.device)