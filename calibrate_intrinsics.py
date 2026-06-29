import cv2
import numpy as np
import glob
import os


# CONFIGURACIÓN DEL TABLERO DE AJEDRE
CHECKERBOARD = (7, 7)

# Tamaño real del lado de un cuadrado en metros (ej. 0.025 para 2.5 cm)
SQUARE_SIZE = 0.05 
# ==============================================================================

def calibrate_camera(image_dir, output_file="camera_intrinsics.npz"):
    # Criterios para refinamiento de esquinas sub-pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Preparar puntos 3D del mundo real
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] # Puntos 3D en el mundo real
    imgpoints = [] # Puntos 2D en el plano de la imagen

    images = glob.glob(os.path.join(image_dir, '*.png'))
    if not images:
        images = glob.glob(os.path.join(image_dir, '*.jpg'))

    if not images:
        print(f"No se encontraron imágenes en {image_dir}")
        return

    print(f"Encontradas {len(images)} imágenes. Detectando tablero...")
    
    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar esquinas con flags para mayor robustez
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if ret:
            objpoints.append(objp)
            # Refinar coordenadas a sub-pixel
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Dibujar y mostrar
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Detectando...', img)
            cv2.waitKey(100)
    
    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        print("\nCalculando calibración intrínseca...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("Calibración exitosa.")
        print("Matriz de Cámara (K):\n", mtx)
        print("Coeficientes de distorsión:\n", dist)
        
        # Calcular error de reproyección (RMSE)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Error de reproyección total (RMSE): {mean_error/len(objpoints)} pixeles")

        # Guardar resultados
        np.savez(output_file, K=mtx, dist=dist)
        print(f"Parámetros guardados en '{output_file}'")
    else:
        print("No se pudo detectar el tablero en ninguna imagen.")

if __name__ == "__main__":
    calibrate_camera("data/images")
