import numpy as np
from PIL import Image

# Função para aplicar o filtro de aguçamento com o operador laplaciano
def agucamento_laplaciano(imagem, kernel):
    # Converter a imagem para escala de cinza
    imagem_cinza = imagem.convert("L")
    pixels = np.array(imagem_cinza)

    # Aplicar o filtro de aguçamento com o operador laplaciano
    pixels_agucados = np.zeros_like(pixels, dtype=np.float32)
    altura, largura = pixels.shape

    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            # Aplicar o kernel aos pixels vizinhos
            vizinhanca = pixels[y - 1: y + 2, x - 1: x + 2]
            valor_agucado = np.sum(kernel * vizinhanca)
            pixels_agucados[y, x] = valor_agucado

    # Ajustar o contraste adicionando o resultado ao original
    imagem_agucada = np.clip(pixels + pixels_agucados, 0, 255).astype(np.uint8)
    imagem_agucada = Image.fromarray(imagem_agucada)

    return imagem_agucada

# Carregar a imagem
caminho_imagem = "11_test.png"
imagem = Image.open(caminho_imagem)

# Definir os kernels
kernel_a = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel_b = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# Aplicar o filtro de aguçamento com o operador laplaciano
imagem_agucada_a = agucamento_laplaciano(imagem, kernel_a)
imagem_agucada_b = agucamento_laplaciano(imagem, kernel_b)

# Salvar as imagens resultantes
imagem_agucada_a.save("kernelA.png")
imagem_agucada_b.save("kernelB.png")

