import cv2
import numpy as np
import socket
import select
import time

# Constantes del Livox
# Tamaño de LivoxLidarCartesianHighRawPoint: int32_t (4) * 3 + uint32_t (4) = 16 bytes
POINT_SIZE = 16 
MAX_UDP_PACKET_SIZE = 65535
TARGET_POINTS_PER_FRAME = 15000 # Cuántos puntos acumular antes de actualizar la pantalla

def load_calibration():
    # Intrínsecos
    try:
        data_int = np.load('camera_intrinsics.npz')
        K = data_int['K']
        dist = data_int['dist']
    except Exception as e:
        print("Error al cargar intrinsics:", e)
        return None, None, None, None

    # Extrínsecos
    try:
        data_ext = np.load('camera_extrinsics.npz')
        R = data_ext['R']
        T = data_ext['T'] # [3, 1]
    except Exception as e:
        print("Error al cargar extrinsics:", e)
        return None, None, None, None

    return K, dist, R, T

def project_points(points_3d, ref_3d, K, dist, R, T, img_width, img_height):
    # points_3d tiene forma (N, 3)
    # Pasar a coordenadas de la cámara
    points_cam = (R @ points_3d.T).T + T.T

    # Filtrar puntos que están detrás de la cámara
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    ref_3d = ref_3d[valid_mask]
    
    # Proyectar
    # OpenCV projectPoints también se puede usar, pero lo hacemos a mano para mayor control o
    # usamos projectPoints directo:
    points_2d, _ = cv2.projectPoints(points_cam, np.zeros((3,1)), np.zeros((3,1)), K, dist)
    points_2d = points_2d.reshape(-1, 2)

    # Filtrar puntos fuera de la imagen
    u = points_2d[:, 0]
    v = points_2d[:, 1]
    inside_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

    u = u[inside_mask].astype(int)
    v = v[inside_mask].astype(int)
    
    # Recuperar el Z (distancia) para colorear
    z = points_cam[:, 2][inside_mask]
    r = ref_3d[inside_mask]

    return u, v, z, r

def main():
    K, dist, R, T = load_calibration()
    if K is None:
        return

    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer la cámara.")
        return
    height, width, _ = frame.shape

    # Configurar socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 50000))
    sock.setblocking(False)

    print("Escuchando puntos Livox en UDP 50000...")
    print("Presiona 'Q' para salir.")

    point_buffer = bytearray()
    
    # Colormap para profundidad
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

    while True:
        # 1. Drenar todos los paquetes UDP disponibles
        while True:
            ready = select.select([sock], [], [], 0.0)
            if ready[0]:
                data, _ = sock.recvfrom(MAX_UDP_PACKET_SIZE)
                point_buffer.extend(data)
            else:
                break

        # 2. Si tenemos suficientes puntos, procesar frame
        num_points = len(point_buffer) // POINT_SIZE
        if num_points >= TARGET_POINTS_PER_FRAME:
            # Capturar cámara
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extraer puntos del buffer
            buffer_to_process = point_buffer[:num_points * POINT_SIZE]
            point_buffer = point_buffer[num_points * POINT_SIZE:] # Limpiar buffer

            # Convertir a numpy
            dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('ref', 'u4')])
            pts = np.frombuffer(buffer_to_process, dtype=dt)

            # Extraer X, Y, Z y convertir a metros (el SDK los da en mm)
            x = pts['x'].astype(np.float32) / 1000.0
            y = pts['y'].astype(np.float32) / 1000.0
            z = pts['z'].astype(np.float32) / 1000.0
            ref = pts['ref'].astype(np.float32)
            points_3d = np.vstack((x, y, z)).T
            
            # Limpiar puntos en 0,0,0 (ruido/origen)
            dists = np.linalg.norm(points_3d, axis=1)
            valid = dists > 0.1
            points_3d = points_3d[valid]
            ref = ref[valid]

            if len(points_3d) > 0:
                # Proyectar
                u, v, z_dist, r_val = project_points(points_3d, ref, K, dist, R, T, width, height)

                # Generar imagen pura del LiDAR (Depth map)
                depth_img = np.zeros_like(frame)

                # Colorear según Reflectividad (contraste dinámico por frame)
                if r_val.max() > r_val.min():
                    r_norm = (r_val - r_val.min()) / (r_val.max() - r_val.min())
                else:
                    r_norm = np.zeros_like(r_val)
                    
                # Aplicamos curva para resaltar blancos y negros
                r_norm = np.power(r_norm, 0.5)
                color_indices = (r_norm * 255).astype(int)
                
                # Dibujar sobre cámara (Fusion) y sobre mapa de profundidad
                # Puntos más grandes (radio 3) para ver el tablero más claro
                for i in range(len(u)):
                    val = int(color_indices[i])
                    color = (val, val, val)
                    cv2.circle(frame, (u[i], v[i]), 3, color, -1)
                    cv2.circle(depth_img, (u[i], v[i]), 3, color, -1)
                
                # --- NUEVO: Zoom Digital ---
                # Recortamos el 50% central de las imágenes y las agrandamos (Zoom x2)
                ch, cw = height // 4, width // 4
                
                def apply_zoom(img):
                    cropped = img[ch:height-ch, cw:width-cw]
                    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

                cam_zoomed = apply_zoom(cap.read()[1] if cap.isOpened() else frame)
                depth_zoomed = apply_zoom(depth_img)
                fusion_zoomed = apply_zoom(frame)

                cv2.imshow("1. Camara Original (Zoom x2)", cam_zoomed)
                cv2.imshow("2. LiDAR (Profundidad Zoom x2)", depth_zoomed)
                cv2.imshow("3. Fusion (Lidar + Camara Zoom x2)", fusion_zoomed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()
