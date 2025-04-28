from PIL import Image
import time
import numpy as np
from image import Imagen
from kernels import Kernels


def main():
    # Generar kernels
    gaussian_kernel = Kernels.generate_gaussian_kernel(21, 4.5)
    sobel_x_kernel = Kernels.generate_sobel_x_kernel(9)
    laplacian_kernel = Kernels.generate_laplacian_kernel(9)

    # Cargar imagen
    image_path = "C:/Users/Anthony/Desktop/Entornos Virtuales Python/env-Pycuda/src/img/ej1.jpg"
    try:
        image = Image.open(image_path)
    except IOError:
        raise Exception(f"Error al cargar la imagen: {image_path}")

    print(f"Imagen cargada con éxito: {image_path}")

    imagen = Imagen()

    # Ejecución Secuencial
    '''

    start_time1 = time.time()
    result_image1 = imagen.apply_convolution_rgb(image, gaussian_kernel)
    end_time1 = time.time()
    duration1 = (end_time1 - start_time1) * 1000  # Tiempo en milisegundos
    print(f"Tiempo de ejecución: {duration1:.2f} ms")
    imagen.save_image(result_image1, "ej1.jpg", "gaussian", 13, "Secuencial")
    '''


    # Ejecución Paralela
    start_time2 = time.time()
    result_image2 = imagen.apply_convolution_parallel_rgb(image, gaussian_kernel)
    end_time2 = time.time()
    duration2 = (end_time2 - start_time2) * 1000  # Tiempo en milisegundos
    print(f"Tiempo de ejecución paralelo: {duration2:.2f} ms")
    imagen.save_image(result_image2, "ej1.jpg", "gaussian", 21, "Paralela")


if __name__ == "__main__":
    main()
