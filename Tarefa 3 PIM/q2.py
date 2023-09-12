from skimage.morphology import binary_erosion, binary_dilation
import numpy as np
from colorama import init, Fore
from PIL import Image

# Cria uma matriz 8x12 com os valores fornecidos
matrizW = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Cria o elemento estruturante B
matrizB = np.array([[0, 1, 0],
                    [1, 1, 0],
                    [0, 1, 0]])

# Define a semente X0 como a erosão binária de W usando B
Xk = binary_erosion(matrizW, matrizB)

# Loop para realizar a dilatação binária até que não haja mais mudanças
while True:
    Xk_1 = Xk
    Xk = binary_dilation(Xk_1, matrizB) & matrizW
    if np.array_equal(Xk, Xk_1):
        break

init(autoreset=True)

for linha in Xk:
    for elemento in linha:
        if elemento == 1:
            print(Fore.GREEN + str(elemento), end=' ')
        else:
            print(elemento, end=' ')
    print()  

matriz = Xk.astype(np.uint8)
matriz = matriz * 255
imagem = Image.fromarray(matriz)
imagem.save('q2.png')