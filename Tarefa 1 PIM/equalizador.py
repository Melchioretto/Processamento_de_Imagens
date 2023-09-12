import cv2
import numpy as np
from skimage import  color, exposure
from PIL import Image
import matplotlib.pyplot as plt
#Essa função calcula o histograma de uma imagem
def meu_histograma(img):
    histograma = np.zeros(256, dtype=np.int32)  # cria um array de 256 elementos com valores inteiros iniciando em 0
    for i in range(img.shape[0]):  # percorre as linhas da imagem
        for j in range(img.shape[1]):  # percorre as colunas da imagem
            intensidade = img[i, j]  # obtém a intensidade do pixel (0-255)
            histograma[intensidade] += 1  # incrementa a contagem de pixels com essa intensidade
    return histograma  # retorna o histograma, um array com a contagem de pixels para cada intensidade

#Essa função calcula a média, variância e entropia de uma imagem.
def calcular_med_var_entropia_imagem(nome_arquivo):
    img = Image.open(nome_arquivo)  # abre a imagem do arquivo
    img_array = np.array(img)  # converte a imagem para um array NumPy
    media = np.mean(img_array)  # calcula a média das intensidades dos pixels na imagem
    variancia = np.var(img_array)  # calcula a variância das intensidades dos pixels na imagem
    histograma = meu_histograma(img_array) / float(img_array.size)  # calcula o histograma normalizado
    entropia = -np.sum(histograma * np.log2(histograma + (histograma == 0)))  # calcula a entropia da imagem
    print("Média: ", media)  # exibe a média das intensidades dos pixels na imagem
    print("Variância: ", variancia)  # exibe a variância das intensidades dos pixels na imagem
    print("Entropia: ", entropia)  # exibe a entropia da imagem
    print(histograma)  # exibe o histograma normalizado da imagem
    plt.bar(range(len(histograma)), histograma)  # cria um gráfico de barras do histograma normalizado
    plt.title("Histograma da imagem '"+nome_arquivo+"'")  # define o título do gráfico
    plt.xlabel("Intensidade de pixel")  # define o label do eixo x
    plt.ylabel("Frequência")  # define o label do eixo y
    plt.show()  # exibe o gráfico


def equalizar_histograma(nome_arquivo):
    img = Image.open(nome_arquivo) # Carrega a imagem
    mg_array = np.array(img) # Converte a imagem para um array numpy
    histograma = meu_histograma(mg_array) # Calcula o histograma da imagem
    distribuicao_acumulada = np.cumsum(histograma) # Calcula a distribuição acumulada do histograma
    distribuicao_acumulada = 255 * (distribuicao_acumulada / distribuicao_acumulada[-1]) # Normaliza a distribuição acumulada para o intervalo [0, 255]
    distribuicao_acumulada = distribuicao_acumulada.astype(np.uint8) # Arredonda os valores da distribuição acumulada para inteiros
    img_equalizada = np.zeros_like(img) # Cria uma matriz de zeros com o mesmo tamanho da imagem original
    for i in range(mg_array.shape[0]): # Percorre as linhas da imagem
        for j in range(mg_array.shape[1]): # Percorre as colunas da imagem
            img_equalizada[i, j] = distribuicao_acumulada[mg_array[i, j]] # Substitui cada pixel da imagem pela intensidade correspondente na distribuição acumulada

    img_equalizada = Image.fromarray(img_equalizada) # Converte a matriz de pixels de volta para uma imagem
    img_equalizada.save("equalizado_"+nome_arquivo) # Salva a imagem equalizada em um arquivo
    plt.imshow(img_equalizada) # Mostra a imagem equalizada
    plt.title("Imagem equalizada") # Define o título do gráfico
    plt.show() # Exibe o gráfico

def equalize_image(nome_arquivo):
    # Carrega a imagem em RGB
    image = cv2.imread(nome_arquivo)
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_yiq = color.rgb2yiq(image) # Converte para YIQ
    y_channel = np.array(image_yiq)[:, :, 0] #separa o y do iq
    i_channel = np.array(image_yiq)[:, :, 1]
    q_channel = np.array(image_yiq)[:, :, 2]
    y_channel_eq = exposure.equalize_hist(y_channel)#equaliza o y
    image_eq = np.dstack((y_channel_eq, i_channel, q_channel))# junta o y com o iq novamente
    image_rgb = color.yiq2rgb(image_eq)# converte para RGB
    image_rgb = (image_rgb * 255).astype(np.uint8) #multiplica a imagem por 255 e converte para inteiro
    Image.fromarray(image_rgb).save('imagem_rgb.jpeg') # Salva a imagem convertida
    
    # Exibe a imagem
    plt.imshow(image_rgb)
    plt.show()


# apenas descomentar e rodar para cada questão *Ctrl + /*
#################### QUESTÃO 1 ##############################

#calcular_med_var_entropia_imagem("equalizado_figuraClara.jpg")
#print("--------------------------")
# calcular_med_var_entropia_imagem("equalizado_figuraEscura.jpg")

#############################################################

#################### QUESTÃO 2 ##############################

# print("\nEqualizando...")
# img_equalizada = equalizar_histograma("figuraClara.jpg")
# img_equalizada = equalizar_histograma("figuraEscura.jpg")
# img_equalizada = equalizar_histograma("marilyn.jpg")
# img_equalizada = equalizar_histograma("xadrez_lowCont.png")
# calcular_med_var_entropia_imagem("marilyn.jpg")
# calcular_med_var_entropia_imagem("xadrez_lowCont.png")
# print("--------------------------")
# print("\nDados dos histogramas após a equalização:\n")
# calcular_med_var_entropia_imagem("equalizado_figuraClara.jpg")
# print("--------------------------")
# calcular_med_var_entropia_imagem("equalizado_figuraEscura.jpg")
# print("--------------------------")
# calcular_med_var_entropia_imagem("equalizado_marilyn.jpg")
# print("--------------------------")
# calcular_med_var_entropia_imagem("equalizado_xadrez_lowCont.png")

#####################################################################

#################### QUESTÃO 3 i ##############################

# img_equalizada = equalizar_histograma("outono_LC.png")
# img_equalizada = equalizar_histograma("predios.jpeg")

#############################################################
#################### QUESTÃO 3 ii #############################
# equalize_image("outono_LC.png")
# equalize_image("predios.jpeg")
#############################################################


