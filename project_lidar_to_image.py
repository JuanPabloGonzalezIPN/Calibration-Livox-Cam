import cv2
import numpy as np
import open3d as o3d
import sys

def project_and_colorize(image_path, pcd_path, intrinsics_path, extrinsics_path, output_pcd="colored_cloud.pcd"):
    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return
    height, width, _ = img.shape

    # Cargar nube de puntos
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_3d = np.asarray(pcd.points)

    # --- NUEVO: Recortar la nube para solucionar el problema del zoom en Open3D ---
    distances = np.linalg.norm(points_3d, axis=1)
    close_mask = distances < 10.0  # Conservamos hasta 10 metros para ver la escena
    
    points_3d = points_3d[close_mask]
    
    # Creamos un nuevo PCD recortado
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Cargar parámetros
    data_int = np.load(intrinsics_path)
    K = data_int['K']
    dist = data_int['dist']

    data_ext = np.load(extrinsics_path)
    R = data_ext['R']
    tvec = data_ext['T']
    
    # Convertir R a vector de rotación rvec para projectPoints
    rvec, _ = cv2.Rodrigues(R)

    # Proyectar puntos 3D al plano de la imagen 2D
    img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    img_points = img_points.reshape(-1, 2)

    # Preparar matriz de colores para la nube
    colors = np.zeros_like(points_3d)

    # Filtrar puntos que caen dentro de la imagen y frente a la cámara (Z > 0)
    # Nota: Transformar puntos al marco de la cámara para checar Z
    points_cam = (R @ points_3d.T).T + tvec.T
    
    for i in range(len(img_points)):
        u, v = int(img_points[i][0]), int(img_points[i][1])
        z = points_cam[i][2]

        if 0 <= u < width and 0 <= v < height and z > 0:
            # OpenCV usa BGR, Open3D usa RGB en rango [0, 1]
            b, g, r = img[v, u]
            colors[i] = [r/255.0, g/255.0, b/255.0]
        else:
            # Puntos fuera de la imagen en gris oscuro
            colors[i] = [0.2, 0.2, 0.2]

    # Asignar colores a la nube
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Guardar la nube coloreada
    o3d.io.write_point_cloud(output_pcd, pcd)
    print(f"Nube de puntos coloreada guardada en: {output_pcd}")

    # Opcional: Visualizar
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: uv run python project_lidar_to_image.py <imagen.png> <nube.pcd>")
    else:
        project_and_colorize(sys.argv[1], sys.argv[2], "camera_intrinsics.npz", "camera_extrinsics.npz")
