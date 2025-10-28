import torch
import numpy as np
import os
from glob import glob
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Importaciones opcionales para visualización
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV no disponible - visualizaciones limitadas")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib no disponible - gráficos limitados")

class VGGTAerialAnalyzer:
    """
    Analizador completo de imágenes aéreas usando el transformador VGGT
    
    VGGT (Visual Geometry Grounded Transformer) funciona de la siguiente manera:
    1. Aggregator: Procesa secuencias de imágenes y extrae tokens de características
    2. Camera Head: Predice parámetros de cámara (pose encoding con 9 dimensiones)
    3. Depth Head: Genera mapas de profundidad densos
    4. Point Head: Predice coordenadas 3D del mundo para cada pixel
    5. Track Head: Rastrea puntos específicos a través de secuencias
    """
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        
    def load_model(self):
        """Carga el modelo preentrenado VGGT"""
        print("Cargando modelo VGGT...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        print("Modelo VGGT cargado exitosamente")
        print(f"    - Aggregator: {self.model.aggregator is not None}")
        print(f"    - Camera Head: {self.model.camera_head is not None}")
        print(f"    - Depth Head: {self.model.depth_head is not None}")
        print(f"    - Point Head: {self.model.point_head is not None}")
        print(f"    - Track Head: {self.model.track_head is not None}")

    def analyze_camera_parameters(self, extrinsic_matrix, intrinsic_matrix, image_shape):
        """
        Analiza parámetros de cámara para extraer información aérea completa
        
        Args:
            extrinsic_matrix: Matriz 3x4 [R|t] - transformación cámara->mundo
            intrinsic_matrix: Matriz 3x3 con parámetros internos de la cámara
            image_shape: (H, W) dimensiones de la imagen
        """
        H, W = image_shape
        
        # Extraer matriz de rotación y vector de traslación
        R = extrinsic_matrix[:3, :3]  # Matriz de rotación 3x3
        t = extrinsic_matrix[:3, 3]   # Vector de traslación 3x1
        
        # Calcular posición de la cámara en coordenadas del mundo
        camera_position = -R.T @ t
        
        # Extraer ángulos de Euler (roll, pitch, yaw)
        sy = torch.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            roll = torch.atan2(R[2,1], R[2,2])
            pitch = torch.atan2(-R[2,0], sy)
            yaw = torch.atan2(R[1,0], R[0,0])
        else:
            roll = torch.atan2(-R[1,2], R[1,1])
            pitch = torch.atan2(-R[2,0], sy)
            yaw = 0
        
        # Convertir a grados
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
        yaw_deg = torch.rad2deg(yaw)
        
        # Calcular altura estimada
        altura_estimada = camera_position[2]
        
        # Clasificar tipo de captura
        pitch_abs = torch.abs(pitch_deg)
        if pitch_abs < 10:
            tipo_captura = "Nadir (Cenital)"
            oblicuidad = "Muy Baja"
        elif pitch_abs < 30:
            tipo_captura = "Oblicua Baja"
            oblicuidad = "Baja"
        elif pitch_abs < 60:
            tipo_captura = "Oblicua Alta"
            oblicuidad = "Alta"
        else:
            tipo_captura = "Lateral"
            oblicuidad = "Muy Alta"
        
        # Parámetros intrínsecos
        fx = intrinsic_matrix[0, 0]  # Distancia focal X
        fy = intrinsic_matrix[1, 1]  # Distancia focal Y
        cx = intrinsic_matrix[0, 2]  # Centro principal X
        cy = intrinsic_matrix[1, 2]  # Centro principal Y
        
        # Calcular FOV
        fov_horizontal = 2 * torch.atan(W / (2 * fx)) * 180 / np.pi
        fov_vertical = 2 * torch.atan(H / (2 * fy)) * 180 / np.pi
        
        # Calcular GSD aproximado (Ground Sample Distance)
        # GSD = (altura * tamaño_pixel) / distancia_focal
        if altura_estimada > 0:
            pixel_size = 0.00000465  # Tamaño típico de pixel en metros (4.65 μm)
            gsd_x = (altura_estimada * pixel_size) / (fx * pixel_size)
            gsd_y = (altura_estimada * pixel_size) / (fy * pixel_size)
        else:
            gsd_x = gsd_y = 0
        
        return {
            'posicion_camara': {
                'x_metros': float(camera_position[0]),
                'y_metros': float(camera_position[1]),
                'altura_estimada_metros': float(altura_estimada),
                'descripcion': f"Cámara a {float(altura_estimada):.1f}m de altura"
            },
            'orientacion': {
                'roll_grados': float(roll_deg),
                'pitch_grados': float(pitch_deg),
                'yaw_grados': float(yaw_deg),
                'descripcion': f"Inclinación: {float(pitch_abs):.1f}°"
            },
            'tipo_captura': {
                'tipo': tipo_captura,
                'oblicuidad': oblicuidad,
                'angulo_inclinacion': float(pitch_abs),
                'es_nadir': pitch_abs < 10
            },
            'parametros_intrínsecos': {
                'fx_pixels': float(fx),
                'fy_pixels': float(fy),
                'cx_pixels': float(cx),
                'cy_pixels': float(cy),
                'fov_horizontal_grados': float(fov_horizontal),
                'fov_vertical_grados': float(fov_vertical),
                'resolucion_imagen': (H, W)
            },
            'parametros_extrinsecos': {
                'matriz_rotacion': R.cpu().numpy(),
                'vector_traslacion': t.cpu().numpy(),
                'matriz_completa': extrinsic_matrix.cpu().numpy()
            },
            'resolucion_espacial': {
                'gsd_x_metros_por_pixel': float(gsd_x),
                'gsd_y_metros_por_pixel': float(gsd_y),
                'descripcion': f"Resolución: {float(gsd_x):.3f} m/pixel"
            }
        }

    def process_depth_map(self, depth_map, depth_conf, threshold=0.5):
        """
        Procesa el mapa de profundidad generado por VGGT
        
        Args:
            depth_map: Tensor con profundidades [H, W]
            depth_conf: Tensor con confianzas [H, W]
            threshold: Umbral de confianza mínima
        """
        # Filtrar por confianza
        valid_mask = depth_conf > threshold
        filtered_depth = depth_map * valid_mask
        
        # Estadísticas
        valid_depths = filtered_depth[valid_mask]
        if len(valid_depths) > 0:
            depth_stats = {
                'profundidad_min': float(valid_depths.min()),
                'profundidad_max': float(valid_depths.max()),
                'profundidad_media': float(valid_depths.mean()),
                'profundidad_std': float(valid_depths.std()),
                'pixels_validos': int(valid_mask.sum()),
                'porcentaje_cobertura': float(valid_mask.sum()) / valid_mask.numel() * 100
            }
        else:
            depth_stats = {
                'profundidad_min': 0,
                'profundidad_max': 0,
                'profundidad_media': 0,
                'profundidad_std': 0,
                'pixels_validos': 0,
                'porcentaje_cobertura': 0
            }
        
        return {
            'mapa_profundidad': filtered_depth,
            'mapa_confianza': depth_conf,
            'mascara_valida': valid_mask,
            'estadisticas': depth_stats
        }

    def analyze_single_image(self, image_path, save_visualizations=True):
        """
        Análisis completo de una imagen aérea individual
        
        Args:
            image_path: Ruta a la imagen
            save_visualizations: Si guardar visualizaciones
        """
        print(f"\nAnalizando: {os.path.basename(image_path)}")
        
        try:
            # Cargar y preprocesar imagen
            images = load_and_preprocess_images([image_path]).to(self.device)
            
            # VGGT espera formato (B, S, C, H, W)
            if len(images.shape) == 4:
                images = images.unsqueeze(1)
            
            print(f" Dimensiones: {images.shape}")
            H, W = images.shape[-2:]
            
            # Ejecutar modelo VGGT
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                    predictions = self.model(images)
            
            results = {
                'archivo': os.path.basename(image_path),
                'ruta_completa': image_path,
                'dimensiones_imagen': (H, W)
            }
            
            # Analizar parámetros de cámara
            if 'pose_enc' in predictions:
                pose_enc = predictions['pose_enc']
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))
                
                # Extraer matrices para análisis
                extrinsic_clean = extrinsic.squeeze(0).squeeze(0)
                intrinsic_clean = intrinsic.squeeze(0).squeeze(0)
                
                camera_analysis = self.analyze_camera_parameters(
                    extrinsic_clean, intrinsic_clean, (H, W)
                )
                results['analisis_camara'] = camera_analysis
            
            # Analizar mapa de profundidad
            if 'depth' in predictions and 'depth_conf' in predictions:
                depth_map = predictions['depth'].squeeze(0).squeeze(0).squeeze(-1)
                depth_conf = predictions['depth_conf'].squeeze(0).squeeze(0)
                
                depth_analysis = self.process_depth_map(depth_map, depth_conf)
                results['analisis_profundidad'] = depth_analysis
            
            # Analizar puntos 3D del mundo
            if 'world_points' in predictions and 'world_points_conf' in predictions:
                world_points = predictions['world_points'].squeeze(0).squeeze(0)
                world_points_conf = predictions['world_points_conf'].squeeze(0).squeeze(0)
                
                results['puntos_3d'] = {
                    'coordenadas_mundo': world_points,
                    'confianza': world_points_conf,
                    'forma': world_points.shape
                }
            
            # Guardar visualizaciones si se solicita
            if save_visualizations:
                self.save_visualizations(results, image_path)
            
            return results
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_visualizations(self, results, image_path):
        """Guarda visualizaciones de los resultados"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"resultados_{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar información en texto
        info_file = os.path.join(output_dir, f"{base_name}_analisis.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            self.write_analysis_report(f, results)
        
        print(f"  Resultados guardados en: {output_dir}")
    
    def write_analysis_report(self, file, results):
        """Escribe reporte detallado del análisis"""
        file.write("="*80 + "\n")
        file.write(f"ANÁLISIS COMPLETO DE IMAGEN AÉREA CON VGGT\n")
        file.write(f"Archivo: {results['archivo']}\n")
        file.write("="*80 + "\n\n")
        
        if 'analisis_camara' in results:
            cam = results['analisis_camara']
            
            file.write("POSICIÓN Y ALTURA DE LA CÁMARA:\n")
            file.write(f"   X: {cam['posicion_camara']['x_metros']:.2f} metros\n")
            file.write(f"   Y: {cam['posicion_camara']['y_metros']:.2f} metros\n")
            file.write(f"   Altura estimada: {cam['posicion_camara']['altura_estimada_metros']:.2f} metros\n")
            file.write(f"   {cam['posicion_camara']['descripcion']}\n\n")
            
            file.write("ORIENTACIÓN Y ÁNGULOS DE CÁMARA:\n")
            file.write(f"   Roll (balanceo): {cam['orientacion']['roll_grados']:.2f}°\n")
            file.write(f"   Pitch (cabeceo): {cam['orientacion']['pitch_grados']:.2f}°\n")
            file.write(f"   Yaw (guiñada): {cam['orientacion']['yaw_grados']:.2f}°\n")
            file.write(f"   {cam['orientacion']['descripcion']}\n\n")
            
            file.write("TIPO DE CAPTURA AÉREA:\n")
            file.write(f"   Tipo: {cam['tipo_captura']['tipo']}\n")
            file.write(f"   Oblicuidad: {cam['tipo_captura']['oblicuidad']}\n")
            file.write(f"   Ángulo de inclinación: {cam['tipo_captura']['angulo_inclinacion']:.2f}°\n")
            file.write(f"   Es Nadir: {'Sí' if cam['tipo_captura']['es_nadir'] else 'No'}\n\n")
            
            file.write("PARÁMETROS INTRÍNSECOS DE CÁMARA:\n")
            intr = cam['parametros_intrínsecos']
            file.write(f"   Distancia focal X: {intr['fx_pixels']:.2f} pixels\n")
            file.write(f"   Distancia focal Y: {intr['fy_pixels']:.2f} pixels\n")
            file.write(f"   Centro principal: ({intr['cx_pixels']:.2f}, {intr['cy_pixels']:.2f}) pixels\n")
            file.write(f"   FOV Horizontal: {intr['fov_horizontal_grados']:.2f}°\n")
            file.write(f"   FOV Vertical: {intr['fov_vertical_grados']:.2f}°\n")
            file.write(f"   Resolución: {intr['resolucion_imagen']}\n\n")
            
            file.write("RESOLUCIÓN ESPACIAL (GSD):\n")
            gsd = cam['resolucion_espacial']
            file.write(f"   GSD X: {gsd['gsd_x_metros_por_pixel']:.4f} metros/pixel\n")
            file.write(f"   GSD Y: {gsd['gsd_y_metros_por_pixel']:.4f} metros/pixel\n")
            file.write(f"   {gsd['descripcion']}\n\n")
        
        if 'analisis_profundidad' in results:
            depth = results['analisis_profundidad']['estadisticas']
            file.write("ANÁLISIS DE MAPA DE PROFUNDIDAD:\n")
            file.write(f"   Profundidad mínima: {depth['profundidad_min']:.2f} metros\n")
            file.write(f"   Profundidad máxima: {depth['profundidad_max']:.2f} metros\n")
            file.write(f"   Profundidad media: {depth['profundidad_media']:.2f} metros\n")
            file.write(f"   Desviación estándar: {depth['profundidad_std']:.2f} metros\n")
            file.write(f"   Cobertura válida: {depth['porcentaje_cobertura']:.1f}%\n")
            file.write(f"   Pixels válidos: {depth['pixels_validos']:,}\n\n")

def print_detailed_analysis(results):
    """Imprime análisis detallado en consola"""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS COMPLETO DE IMAGEN AÉREA: {results['archivo']}")
    print(f"{'='*80}")
    
    if 'analisis_camara' in results:
        cam = results['analisis_camara']
        
        print(f"\nINFORMACIÓN DE VUELO Y CÁMARA:")
        print(f"   Posición estimada: ({cam['posicion_camara']['x_metros']:.1f}, {cam['posicion_camara']['y_metros']:.1f}) m")
        print(f"   Altura de vuelo: {cam['posicion_camara']['altura_estimada_metros']:.1f} metros")
        print(f"   Inclinación: {cam['orientacion']['pitch_grados']:.1f}° (Roll: {cam['orientacion']['roll_grados']:.1f}°, Yaw: {cam['orientacion']['yaw_grados']:.1f}°)")
        print(f"   Tipo de captura: {cam['tipo_captura']['tipo']} - {cam['tipo_captura']['oblicuidad']}")
        print(f"   Campo de visión: {cam['parametros_intrínsecos']['fov_horizontal_grados']:.1f}° × {cam['parametros_intrínsecos']['fov_vertical_grados']:.1f}°")
        print(f"   Resolución espacial: {cam['resolucion_espacial']['gsd_x_metros_por_pixel']:.3f} m/pixel")
    
    if 'analisis_profundidad' in results:
        depth_stats = results['analisis_profundidad']['estadisticas']
        print(f"\nMAPA DE PROFUNDIDAD:")
        print(f"   Rango: {depth_stats['profundidad_min']:.1f} - {depth_stats['profundidad_max']:.1f} metros")
        print(f"   Profundidad media: {depth_stats['profundidad_media']:.1f} ± {depth_stats['profundidad_std']:.1f} metros")
        print(f"   Cobertura válida: {depth_stats['porcentaje_cobertura']:.1f}%")

# ============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================================================

def main():
    """Función principal para análisis de imágenes aéreas"""
    
    print("ANALIZADOR AVANZADO DE IMÁGENES AÉREAS CON VGGT")
    print("="*60)
    
    # Configuración
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Dispositivo: {device}")
    print(f"Tipo de datos: {dtype}")
    
    # Inicializar analizador
    analyzer = VGGTAerialAnalyzer(device=device, dtype=dtype)
    analyzer.load_model()
    
    # Configurar imágenes a procesar
    image_folder = "C:/Users/pedroam/Documents/Datasets/VisDrone/img/train/test"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_names = []
    
    for extension in image_extensions:
        image_names.extend(glob(os.path.join(image_folder, extension)))
        image_names.extend(glob(os.path.join(image_folder, extension.upper())))
    
    # Procesar imágenes (limitar a 3 para el ejemplo)
    image_names = sorted(image_names)[:3]
    
    print(f"\nEncontradas {len(image_names)} imágenes para análisis completo")
    
    all_results = []
    
    for i, image_path in enumerate(image_names, 1):
        print(f"\n{'='*60}")
        print(f"PROCESANDO IMAGEN {i}/{len(image_names)}")
        print(f"{'='*60}")
        
        # Analizar imagen
        results = analyzer.analyze_single_image(image_path, save_visualizations=True)
        
        if results:
            all_results.append(results)
            print_detailed_analysis(results)
        
        print(f"\nImagen {i} completada")
    
    # Resumen final
    print(f"\nANÁLISIS COMPLETADO")
    print(f"   Total de imágenes procesadas: {len(all_results)}")
    print(f"   Archivos de resultados guardados en carpetas individuales")
    
    # Estadísticas generales
    if all_results:
        alturas = []
        tipos_captura = []
        
        for result in all_results:
            if 'analisis_camara' in result:
                alturas.append(result['analisis_camara']['posicion_camara']['altura_estimada_metros'])
                tipos_captura.append(result['analisis_camara']['tipo_captura']['tipo'])
        
        if alturas:
            print(f"\nESTADÍSTICAS DEL DATASET:")
            print(f"   Altura promedio: {np.mean(alturas):.1f} metros")
            print(f"   Rango de alturas: {min(alturas):.1f} - {max(alturas):.1f} metros")
            print(f"   Tipos de captura encontrados: {set(tipos_captura)}")

if __name__ == "__main__":
    main()

def print_aerial_analysis(info, image_name):
    """
    Imprime el análisis de imagen aérea de forma organizada
    """
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE IMAGEN AÉREA: {os.path.basename(image_name)}")
    print(f"{'='*80}")
    
    # Información de posición
    print(f"\nPOSICIÓN DE LA CÁMARA:")
    print(f"   X: {info['posicion']['x']:.2f} m")
    print(f"   Y: {info['posicion']['y']:.2f} m") 
    print(f"   Altura (Z): {info['posicion']['z_altura']:.2f} m")
    
    # Información de orientación
    print(f"\nORIENTACIÓN DE LA CÁMARA:")
    print(f"   Roll:  {info['angulos']['roll_deg']:.2f}°")
    print(f"   Pitch: {info['angulos']['pitch_deg']:.2f}°")
    print(f"   Yaw:   {info['angulos']['yaw_deg']:.2f}°")
    
    # Tipo de captura
    print(f"\nTIPO DE CAPTURA:")
    print(f"   Tipo: {info['tipo_captura']}")
    print(f"   Oblicuidad: {info['oblicuidad']}")
    
    # Parámetros de cámara
    print(f"\nPARÁMETROS DE CÁMARA:")
    print(f"   Distancia focal X: {info['parametros_intrinsecos']['fx']:.2f} px")
    print(f"   Distancia focal Y: {info['parametros_intrinsecos']['fy']:.2f} px")
    print(f"   Centro principal: ({info['parametros_intrinsecos']['cx']:.2f}, {info['parametros_intrinsecos']['cy']:.2f}) px")
    print(f"   FOV Horizontal: {info['parametros_intrinsecos']['fov_horizontal_deg']:.2f}°")
    print(f"   FOV Vertical: {info['parametros_intrinsecos']['fov_vertical_deg']:.2f}°")

# ============================================================================
# CONFIGURACIÓN Y INICIALIZACIÓN
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("ANALIZADOR DE IMÁGENES AÉREAS CON VGGT")
print("="*60)
print(f"Dispositivo: {device}")
print(f"Tipo de datos: {dtype}")

# Cargar modelo
print("\nCargando modelo VGGT...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Modelo cargado exitosamente")

# El código principal está ahora en la función main() al final del archivo
