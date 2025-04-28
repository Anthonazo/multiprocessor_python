import pycuda.autoinit
import numpy as np
from PIL import Image
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os

class Imagen:
    def __init__(self):
        self.image = None

    def save_image(self, image, original_name, filter_name, kernel_size, execution_type):
        try:
            base_name = os.path.splitext(original_name)[0]
            file_name = f"{base_name}_{filter_name}_k{kernel_size}_{execution_type}.png"
            output_path = os.path.join("img", file_name)

            # Crear la carpeta si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            image.save(output_path, "PNG")
            print(f"Imagen guardada en: {output_path}")
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")

    def apply_convolution_rgb(self, image, kernel):
        width, height = image.size
        size_kernel = len(kernel)
        margin = size_kernel // 2

        result_image = Image.new("RGB", (width, height))
        pixels = result_image.load()

        for x in range(margin, width - margin):
            for y in range(margin, height - margin):
                red_sum = green_sum = blue_sum = 0

                for i in range(size_kernel):
                    for j in range(size_kernel):
                        pixel = image.getpixel((x + i - margin, y + j - margin))
                        red, green, blue = pixel

                        red_sum += red * kernel[i][j]
                        green_sum += green * kernel[i][j]
                        blue_sum += blue * kernel[i][j]

                new_red = min(max(int(red_sum), 0), 255)
                new_green = min(max(int(green_sum), 0), 255)
                new_blue = min(max(int(blue_sum), 0), 255)

                pixels[x, y] = (new_red, new_green, new_blue)

        return result_image

    def apply_convolution_parallel_rgb(self, image, kernel):
        width, height = image.size
        size_kernel = len(kernel)
        margin = size_kernel // 2

        # Convertir la imagen a un arreglo numpy
        img_array = np.array(image).astype(np.float32)  # <-- CAMBIO: convertir a float32
        kernel = np.array(kernel, dtype=np.float32)

        # Preparamos el resultado en un arreglo numpy
        result_image = np.zeros_like(img_array)

        # Transferir los datos a la GPU
        img_array_gpu = cuda.to_device(img_array)
        result_image_gpu = cuda.to_device(result_image)
        kernel_gpu = cuda.to_device(kernel)

        # Definir el tamaño del bloque y la cuadrícula
        block_size = (16, 16, 1)  # Bloque de 16x16 hilos
        grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)

        # Código CUDA para la convolución
        convolution_kernel = """
        __global__ void apply_convolution(float *img_array, float *kernel, float *result_image, int width, int height, int kernel_size, int margin) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height)
                return;

            if (x >= margin && x < width - margin && y >= margin && y < height - margin) {
                float red_sum = 0.0f;
                float green_sum = 0.0f;
                float blue_sum = 0.0f;

                for (int i = -margin; i <= margin; i++) {
                    for (int j = -margin; j <= margin; j++) {
                        int img_x = x + i;
                        int img_y = y + j;

                        float pixel = img_array[(img_y * width + img_x) * 3];  // Red
                        red_sum += pixel * kernel[(i + margin) * kernel_size + (j + margin)];

                        pixel = img_array[(img_y * width + img_x) * 3 + 1];  // Green
                        green_sum += pixel * kernel[(i + margin) * kernel_size + (j + margin)];

                        pixel = img_array[(img_y * width + img_x) * 3 + 2];  // Blue
                        blue_sum += pixel * kernel[(i + margin) * kernel_size + (j + margin)];
                    }
                }

                int result_index = (y * width + x) * 3;
                result_image[result_index] = fminf(fmaxf(red_sum, 0.0f), 255.0f);
                result_image[result_index + 1] = fminf(fmaxf(green_sum, 0.0f), 255.0f);
                result_image[result_index + 2] = fminf(fmaxf(blue_sum, 0.0f), 255.0f);
            }
        }

        """

        # Compilar el kernel CUDA
        module = SourceModule(convolution_kernel)
        convolution_func = module.get_function("apply_convolution")

        # Llamar al kernel
        convolution_func(img_array_gpu, kernel_gpu, result_image_gpu, np.int32(width), np.int32(height), np.int32(size_kernel), np.int32(margin),
                         block=block_size, grid=grid_size)

        # Copiar el resultado de vuelta a la CPU
        result_array = np.empty((height, width, 3), dtype=np.float32)  # <-- CAMBIO: también float32
        cuda.memcpy_dtoh(result_array, result_image_gpu)
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)  # <-- CAMBIO: convertir de vuelta a uint8
        result_image = Image.fromarray(result_array, 'RGB')


        return result_image
