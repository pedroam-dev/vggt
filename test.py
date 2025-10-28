import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Test CUDA functionality
print("=== CUDA Diagnostic Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("CUDA is not available!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    print(f"Using dtype: {dtype} (Compute Capability: {capability})")
else:
    dtype = torch.float32
    print(f"Using dtype: {dtype} (CPU mode)")

print("\n=== Model Loading ===")
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
try:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("✅ Model loaded successfully")
    
    # Show model memory usage after loading
    if torch.cuda.is_available():
        print(f"Memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("\n=== Image Processing ===")
# Load and preprocess example images (replace with your own image paths)
image_names = ["C:/Users/pedroam/Documents/Datasets/VisDrone/img/train/images/0000002_00005_d_0000014.jpg"]  

try:
    images = load_and_preprocess_images(image_names).to(device)
    print(f"✅ Images loaded successfully")
    print(f"Image tensor shape: {images.shape}")
    print(f"Image tensor dtype: {images.dtype}")
    print(f"Image tensor device: {images.device}")
    
    # Show memory usage after loading images
    if torch.cuda.is_available():
        print(f"Memory after loading images: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"❌ Error loading images: {e}")
    exit(1)

print("\n=== Model Inference ===")
try:
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    
    print("✅ Inference completed successfully")
    print(f"Prediction type: {type(predictions)}")
    
    if hasattr(predictions, 'shape'):
        print(f"Prediction shape: {predictions.shape}")
    elif isinstance(predictions, (list, tuple)):
        print(f"Number of prediction outputs: {len(predictions)}")
        for i, pred in enumerate(predictions):
            if hasattr(pred, 'shape'):
                print(f"  Output {i}: {pred.shape}")
    
    # Final memory usage
    if torch.cuda.is_available():
        print(f"Final memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"❌ Error during inference: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Completed ===")