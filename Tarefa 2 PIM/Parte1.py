import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def filtro_media(imagem, nucleo):
    altura, largura = imagem.shape
    nh, nw = nucleo.shape
    margem_h = nh // 2  # Margem vertical do kernel
    margem_w = nw // 2  # Margem horizontal do kernel
    nova_imagem = np.zeros((altura, largura))
    for i in range(margem_h, altura - margem_h):
        for j in range(margem_w, largura - margem_w):
            nova_imagem[i, j] = np.sum(imagem[i - margem_h:i + margem_h + 1, j - margem_w:j + margem_w + 1] * nucleo)
            # Calcula a média ponderada dos pixels vizinhos e atribui o resultado ao pixel correspondente na nova imagem
    return nova_imagem

# Abre a imagem em tons de cinza
imagem = Image.open('Lua1_gray.jpg')
imagem = imagem.convert('L')
imagem = np.array(imagem)

# Define o kernel de média
nucleo_media = np.ones((3, 3))/9

# Define os valores de sigma para o filtro gaussiano
sigma1 = 1.0
sigma2 = 0.6

def gaussiano_cinza(imagem, sigma):
    altura, largura = imagem.shape
    X = np.array([[-1, 0, 1], 
                  [-1, 0, 1],   
                  [-1, 0, 1]])  # Componente X do kernel de diferenciação
    Y = np.array([[-1, -1, -1], 
                  [ 0,  0,  0],   
                  [ 1,  0,  1]])  # Componente Y do kernel de diferenciação
    nucleo_gaussiano = np.zeros((3, 3))
    for i in range(-1, 2):
        for j in range(-1, 2):
            nucleo_gaussiano[i + 1, j + 1] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(X[i+1, j+1] ** 2 + Y[i+1, j+1] ** 2) / (2 * sigma ** 2))
            # Calcula o valor de cada elemento do kernel gaussiano usando a fórmula do filtro gaussiano
    nucleo_gaussiano /= np.sum(nucleo_gaussiano)  # Normaliza o kernel gaussiano para que a soma de todos os elementos seja 1

    nova_imagem = np.zeros((altura, largura))
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            nova_imagem[i, j] = np.sum(imagem[i - 1:i + 2, j - 1:j + 2] * nucleo_gaussiano)
            # Aplica o kernel gaussiano aos pixels vizinhos e atribui o resultado ao pixel correspondente na nova imagem
    return nova_imagem

# Aplica o filtro de média na imagem
imagem_filtrada_media = filtro_media(imagem, nucleo_media)

# Aplica o filtro gaussiano na imagem usando sigma1
imagem_filtrada_sigma1 = gaussiano_cinza(imagem, sigma1)

# Aplica o filtro gaussiano na imagem usando sigma2
imagem_filtrada_sigma2 = gaussiano_cinza(imagem, sigma2)

# Plota e salva a imagem original
plt.imshow(imagem, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title('Imagem Original')
plt.savefig('imagem_original.png')

# Plota e salva a imagem filtrada com o filtro de média
plt.imshow(imagem_filtrada_media, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title('Imagem Filtrada (Média)')
plt.savefig('imagem_filtrada_media.png')

# Plota e salva a imagem filtrada com o filtro gaussiano (sigma=1.0)
plt.imshow(imagem_filtrada_sigma1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title('Imagem Filtrada (Sigma=1.0)')
plt.savefig('imagem_filtrada_sigma1.png')

# Plota e salva a imagem filtrada com o filtro gaussiano (sigma=0.6)
plt.imshow(imagem_filtrada_sigma2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title('Imagem Filtrada (Sigma=0.6)')
plt.savefig('imagem_filtrada_sigma0.6.png')
