import cv2
import numpy as np
import open3d as o3d
import os

# ==============================================================================
# CONFIGURACIÓN DEL TABLERO
CHECKERBOARD = (7, 7)
SQUARE_SIZE = 0.025
# ==============================================================================

def pick_points_3d(pcd):
    print("---------------------------------------")
    print("Instrucciones para Open3D:")
    print("1. Mantén presionada la tecla [Shift] y haz clic izquierdo para seleccionar puntos.")
    print("2. Selecciona las 4 INTERSECCIONES INTERNAS del ajedrez")
    print("   Los cruces de los cuadros blancos y negros).")
    print("3. Seleccionarlas en el MISMO ORDEN (1, 2, 3, 4) que viste en la imagen de la cámara.")
    print("4. Presiona 'Q' o cierra la ventana cuando termines de elegir los 4 puntos.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def calibrate_extrinsics(image_path, pcd_path, intrinsics_path):
    # Cargar intrínsecos
    data = np.load(intrinsics_path)
    K = data['K']
    dist = data['dist']

    # Cargar Imagen y detectar tablero
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if not ret:
        print("Error: No se pudo detectar el tablero en la imagen.")
        return

    # Refinar esquinas 2D
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_2d = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Extraer las 4 esquinas externas en 2D en el orden correcto
    # Suponiendo que las esquinas devueltas van de izquierda a derecha, arriba a abajo
    top_left = corners_2d[0][0]
    top_right = corners_2d[CHECKERBOARD[0] - 1][0]
    bottom_right = corners_2d[-1][0]
    bottom_left = corners_2d[-CHECKERBOARD[0]][0]

    image_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Mostrar esquinas 2D seleccionadas
    for i, pt in enumerate(image_points):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i+1), (int(pt[0])+10, int(pt[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Esquinas 2D (Verifica el orden 1->2->3->4)", cv2.resize(img, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Cargar Nube de Puntos
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print("Error: Nube de puntos vacía o no encontrada.")
        return

    # --- NUEVO: Recortar la nube para solucionar el problema del zoom en Open3D ---
    # Al eliminar los puntos lejanos (la pared de fondo, techo, etc.), 
    # la caja delimitadora (bounding box) de Open3D se hace pequeña y permite hacer zoom infinito.
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Filtrar puntos a menos de 4 metros del LiDAR
    distances = np.linalg.norm(points, axis=1)
    close_mask = distances < 4.0
    
    pcd_cropped = o3d.geometry.PointCloud()
    pcd_cropped.points = o3d.utility.Vector3dVector(points[close_mask])
    if len(colors) > 0:
        pcd_cropped.colors = o3d.utility.Vector3dVector(colors[close_mask])
    
    print(f"\nRecortando nube de {len(points)} a {len(pcd_cropped.points)} puntos cercanos para facilitar el zoom.")
    
    # Obtener puntos 3D del usuario
    picked_indices = pick_points_3d(pcd_cropped)
    if len(picked_indices) != 4:
        print(f"Error: Debes seleccionar exactamente 4 puntos. Seleccionaste {len(picked_indices)}.")
        return

    # Extraer los puntos del PCD recortado
    points_3d = np.asarray(pcd_cropped.points)
    object_points = points_3d[picked_indices]

    print("\nPuntos 3D seleccionados:")
    print(object_points)

    # Calcular PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        R, _ = cv2.Rodrigues(rvec)
        print("\nCalibración Extrínseca Exitosa")
        print("Matriz de Rotación (R):")
        print(R)
        print("Vector de Traslación (T):")
        print(tvec)

        np.savez("camera_extrinsics.npz", R=R, T=tvec)
        print("Guardado en 'camera_extrinsics.npz'")
    else:
        print("Error: solvePnP falló.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: uv run python calibrate_extrinsics.py <imagen.png> <nube.pcd>")
    else:
        calibrate_extrinsics(sys.argv[1], sys.argv[2], "camera_intrinsics.npz")
