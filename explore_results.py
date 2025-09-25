import torch
import numpy as np
import json
import os
import glob

def explore_vggt_results(results_folder="/Users/pedroam/Documents/vggt_results"):
    """Explora todos los archivos de resultados VGGT"""
    
    # Buscar archivos de resultados
    pt_files = glob.glob(os.path.join(results_folder, "predictions_*.pt"))
    json_files = glob.glob(os.path.join(results_folder, "metadata_*.json"))
    numpy_folders = glob.glob(os.path.join(results_folder, "numpy_*"))
    
    print(f"Archivos encontrados en {results_folder}:")
    print(f"  - {len(pt_files)} archivos de predicciones (.pt)")
    print(f"  - {len(json_files)} archivos de metadatos (.json)")
    print(f"  - {len(numpy_folders)} carpetas de arrays NumPy")
    
    if not pt_files:
        print("No se encontraron archivos de resultados")
        return
    
    # Usar el archivo más reciente
    latest_pt = max(pt_files, key=os.path.getctime)
    timestamp = os.path.basename(latest_pt).replace("predictions_", "").replace(".pt", "")
    
    print(f"\nExplorando resultados más recientes: {timestamp}")
    
    # 1. Cargar predicciones
    print("\n1. Cargando predicciones...")
    predictions = torch.load(latest_pt)
    
    print(f"Tipo: {type(predictions)}")
    print(f"Claves disponibles: {list(predictions.keys())}")
    
    for key, value in predictions.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: forma {value.shape}, tipo {value.dtype}")
            # Mostrar estadísticas básicas
            if value.numel() > 0:
                print(f"    Min: {value.min().item():.4f}, Max: {value.max().item():.4f}, Media: {value.mean().item():.4f}")
        else:
            print(f"  {key}: tipo {type(value)}")
    
    # 2. Cargar metadatos
    metadata_file = os.path.join(results_folder, f"metadata_{timestamp}.json")
    if os.path.exists(metadata_file):
        print("\n2. Cargando metadatos...")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Número de imágenes: {metadata['num_images']}")
        print(f"Dispositivo: {metadata['device']}")
        print(f"Tipo de datos: {metadata['dtype']}")
        print(f"Imágenes procesadas: {metadata['image_names'][:3]}{'...' if len(metadata['image_names']) > 3 else ''}")
    
    # 3. Explorar arrays NumPy
    numpy_folder = os.path.join(results_folder, f"numpy_{timestamp}")
    if os.path.exists(numpy_folder):
        print("\n3. Explorando arrays NumPy...")
        npy_files = glob.glob(os.path.join(numpy_folder, "*.npy"))
        
        for npy_file in npy_files:
            array_name = os.path.basename(npy_file).replace(".npy", "")
            array = np.load(npy_file)
            print(f"  {array_name}: forma {array.shape}, dtype {array.dtype}")
            print(f"    Min: {array.min():.4f}, Max: {array.max():.4f}, Media: {array.mean():.4f}")
    
    return predictions, metadata if 'metadata' in locals() else None

# Ejecutar exploración
if __name__ == "__main__":
    predictions, metadata = explore_vggt_results()