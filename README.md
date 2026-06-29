# Calibración y Fusión: LiDAR Livox Mid-360 + Cámara Web

Este repositorio contiene un sistema completo para calibrar extrínseca e intrínsecamente un LiDAR Livox Mid-360 con una cámara web estándar (RGB), y visualizar la fusión de ambos sensores en tiempo real. 

El proyecto fue construido utilizando **Python (OpenCV, Open3D)** y **C++ (Livox-SDK2)**, operando en Ubuntu 24.04 **sin depender de ROS**.

## Estructura del Proyecto

El sistema está dividido en tres etapas principales. Cada etapa tiene su propia documentación detallada en la carpeta `docs/`:

1. **[Calibración Intrínseca](docs/01_calibracion_intrinseca.md)**: Obtención de la matriz de la cámara y coeficientes de distorsión usando un tablero de ajedrez.
2. **[Calibración Extrínseca](docs/02_calibracion_extrinseca.md)**: Cálculo de la rotación y traslación (R, T) entre el LiDAR y la cámara mediante la correspondencia manual de puntos (2D a 3D).
3. **[Fusión en Tiempo Real](docs/03_fusion_tiempo_real.md)**: Proyección de la nube de puntos 3D sobre el feed de video 2D a alta velocidad mediante un puente UDP.

## Requisitos del Sistema
- Ubuntu 24.04 (Probado y verificado)
- Python 3.11+ (Manejador virtual `uv` recomendado)
- CMake y build-essential
- [Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2)

## Instalación Rápida
1. Clona este repositorio.
2. Crea el entorno virtual e instala dependencias de Python:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install opencv-python open3d numpy scipy
   ```
3. Compila el SDK de Livox (asegúrate de que la IP en `Livox-SDK2/samples/livox_lidar_quick_start/mid360_config.json` coincida con la tuya).
   ```bash
   cd Livox-SDK2/build
   cmake .. && make -j4
   ```

*(Revisa la documentación en `docs/` para guías paso a paso de cada script).*
