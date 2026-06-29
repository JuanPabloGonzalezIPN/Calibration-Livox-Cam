import cv2
import sys

def find_board_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    print(f"Probando diferentes dimensiones de tablero en {image_path}...")
    
    # Probar combinaciones desde 4x3 hasta 15x15
    for w in range(15, 3, -1):
        for h in range(15, 3, -1):
            if w < h: 
                continue # Evitar probar (6,8) si ya probamos (8,6)
                
            ret, corners = cv2.findChessboardCorners(gray, (w, h), flags)
            if ret:
                print("\n" + "="*50)
                print(f"¡ÉXITO! Se detectó un tablero con dimensiones: CHECKERBOARD = ({w}, {h})")
                print("="*50 + "\n")
                
                # Mostrar el resultado para estar seguros
                cv2.drawChessboardCorners(img, (w, h), corners, ret)
                cv2.imshow(f"Tablero {w}x{h} Detectado", cv2.resize(img, (800, 600)))
                print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return (w, h)
                
    print("\nNo se pudo detectar el tablero con ninguna dimensión estándar (entre 4x4 y 15x15).")
    print("Por favor, verifica que la imagen no esté borrosa y que el tablero esté completamente visible.")
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python find_board_size.py <ruta_a_la_imagen.png>")
    else:
        find_board_dimensions(sys.argv[1])
