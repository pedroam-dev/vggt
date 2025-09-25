import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import glob
import json
import numpy as np
from datetime import datetime

# Detectar dispositivo disponible
# Nota: Usamos CPU por compatibilidad con todas las operaciones de VGGT
if torch.cuda.is_available():
    device = "cuda"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
elif torch.backends.mps.is_available():
    # MPS tiene limitaciones con ciertas operaciones de interpolación
    print("Detectado chip M1/M2, pero usando CPU por compatibilidad completa con VGGT")
    device = "cpu"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32  # CPU funciona mejor con float32

print(f"Usando dispositivo: {device} con dtype: {dtype}")

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Especifica la carpeta de imágenes
images_folder = "/Users/pedroam/Pictures/visdrone_muestra"  # Cambia esta ruta por tu carpeta

# Crear carpeta de resultados
results_folder = "/Users/pedroam/Documents/vggt_results"
os.makedirs(results_folder, exist_ok=True)

# Extensiones de imagen soportadas
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

# Buscar todas las imágenes en la carpeta
image_names = []
for extension in image_extensions:
    image_names.extend(glob.glob(os.path.join(images_folder, extension)))
    image_names.extend(glob.glob(os.path.join(images_folder, extension.upper())))

if not image_names:
    print(f"No se encontraron imágenes en la carpeta: {images_folder}")
    print("Asegúrate de que la carpeta existe y contiene imágenes en formato JPG, PNG, etc.")
    exit(1)

print(f"Encontradas {len(image_names)} imágenes:")
for img in image_names[:5]:  # Mostrar solo las primeras 5
    print(f"  - {os.path.basename(img)}")
if len(image_names) > 5:
    print(f"  ... y {len(image_names) - 5} más")

# Load and preprocess images from folder
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    # Usar autocast apropiado según el dispositivo
    if device == "cuda":
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    else:
        # CPU o MPS con fallback
        predictions = model(images.to(dtype))

print("Predicciones completadas exitosamente")
print(f"Procesadas {len(image_names)} imágenes")

# El modelo VGGT devuelve un diccionario con diferentes tipos de predicciones
print(f"Tipo de predicciones: {type(predictions)}")
print(f"Claves disponibles: {list(predictions.keys())}")

# Mostrar información sobre cada tipo de predicción
for key, value in predictions.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: forma {value.shape}, tipo {value.dtype}")
    else:
        print(f"  {key}: tipo {type(value)}")

# Guardar resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Guardar tensores como archivos .pt
predictions_file = os.path.join(results_folder, f"predictions_{timestamp}.pt")
torch.save(predictions, predictions_file)
print(f"Predicciones guardadas en: {predictions_file}")

# 2. Guardar metadatos como JSON
metadata = {
    "timestamp": timestamp,
    "num_images": len(image_names),
    "image_names": [os.path.basename(img) for img in image_names],
    "device": device,
    "dtype": str(dtype),
    "predictions_info": {}
}

for key, value in predictions.items():
    if hasattr(value, 'shape'):
        metadata["predictions_info"][key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype)
        }
    else:
        metadata["predictions_info"][key] = {
            "type": str(type(value))
        }

metadata_file = os.path.join(results_folder, f"metadata_{timestamp}.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadatos guardados en: {metadata_file}")

# 3. Guardar predicciones como numpy arrays (más fácil de leer en otros programas)
numpy_folder = os.path.join(results_folder, f"numpy_{timestamp}")
os.makedirs(numpy_folder, exist_ok=True)

for key, value in predictions.items():
    if hasattr(value, 'cpu'):  # Es un tensor
        numpy_array = value.cpu().numpy()
        numpy_file = os.path.join(numpy_folder, f"{key}.npy")
        np.save(numpy_file, numpy_array)
        print(f"  {key} guardado como: {numpy_file}")

print(f"\nTodos los resultados guardados en: {results_folder}")
print("Archivos generados:")
print(f"  - {os.path.basename(predictions_file)} (predicciones completas)")
print(f"  - {os.path.basename(metadata_file)} (información sobre la ejecución)")
print(f"  - numpy_{timestamp}/ (arrays NumPy individuales)")