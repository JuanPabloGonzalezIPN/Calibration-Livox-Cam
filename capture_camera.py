import cv2
import os

def main():
    # Asegúrate de que el índice (0) corresponde a tu cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Crear carpeta para guardar imágenes
    output_dir = "data/images"
    os.makedirs(output_dir, exist_ok=True)

    print("Cámara iniciada.")
    print("Presiona 's' para guardar un fotograma.")
    print("Presiona 'q' o 'Esc' para salir.")

    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el fotograma")
            break

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = os.path.join(output_dir, f"image_{img_count:04d}.png")
            cv2.imwrite(filename, frame)
            print(f"Fotograma guardado: {filename}")
            img_count += 1
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
