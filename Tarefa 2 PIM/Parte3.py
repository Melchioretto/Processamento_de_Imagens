import cv2
import numpy as np

#Sobel calcula o gradiente da intensidade da imagem em cada ponto, dando a direcção da maior variação de claro
#para escuro e a quantidade de variação nessa direcção. 
#Assim, obtém-se uma noção de como varia a luminosidade em cada ponto, de forma mais suave ou abrupta.


#Prewitt é uma operador de diferença, calcular uma aproximação do gradiente da imagem de intensidade de função



# Função para aplicar o operador gradiente utilizando uma máscara
def aplicar_operador_gradiente(imagem, mascara):
    linhas, colunas = imagem.shape
    linhas_mascara, colunas_mascara = mascara.shape
    metade_linhas_mascara = linhas_mascara // 2
    metade_colunas_mascara = colunas_mascara // 2

    gradiente = np.zeros_like(imagem, dtype=np.float32)

    for i in range(metade_linhas_mascara, linhas - metade_linhas_mascara):
        for j in range(metade_colunas_mascara, colunas - metade_colunas_mascara):
            janela = imagem[i - metade_linhas_mascara : i + metade_linhas_mascara + 1, j - metade_colunas_mascara : j + metade_colunas_mascara + 1]
            gradiente[i, j] = np.sum(janela * mascara)

    return gradiente

# Função para calcular a magnitude do gradiente
def calcular_magnitude(gradiente_x, gradiente_y):
    magnitude = np.sqrt(gradiente_x**2 + gradiente_y**2)
    return magnitude

# Carregando a imagem em tons de cinza
imagem = cv2.imread('chessboard_inv.png', cv2.IMREAD_GRAYSCALE)
#imagem = cv2.imread('Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)
# Máscaras de Sobel
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# Máscaras de Prewitt
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)

# Máscaras de Scharr
scharr_x = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float32)

scharr_y = np.array([[-3, -10, -3],
                     [0, 0, 0],
                     [3, 10, 3]], dtype=np.float32)

# Aplicando o operador gradiente com as máscaras de Sobel
gradiente_sobel_x = aplicar_operador_gradiente(imagem, sobel_x)
gradiente_sobel_y = aplicar_operador_gradiente(imagem, sobel_y)

# Aplicando o operador gradiente com as máscaras de Prewitt
gradiente_prewitt_x = aplicar_operador_gradiente(imagem, prewitt_x)
gradiente_prewitt_y = aplicar_operador_gradiente(imagem, prewitt_y)

# Aplicando o operador gradiente com as máscaras de Scharr
gradiente_scharr_x = aplicar_operador_gradiente(imagem, scharr_x)
gradiente_scharr_y = aplicar_operador_gradiente(imagem, scharr_y)

# Função para calcular a magnitude do gradiente
def calcular_magnitude(gradiente_x, gradiente_y):
    magnitude = np.sqrt(gradiente_x**2 + gradiente_y**2)
    return magnitude

# Calculando a magnitude do gradiente
magnitude_sobel = calcular_magnitude(gradiente_sobel_x, gradiente_sobel_y)
magnitude_prewitt = calcular_magnitude(gradiente_prewitt_x, gradiente_prewitt_y)
magnitude_scharr = calcular_magnitude(gradiente_scharr_x, gradiente_scharr_y)

# Salvando as imagens resultantes da magnitude
cv2.imwrite('magnitude_sobel.jpg', magnitude_sobel)
cv2.imwrite('magnitude_prewitt.jpg', magnitude_prewitt)
cv2.imwrite('magnitude_scharr.jpg', magnitude_scharr)
