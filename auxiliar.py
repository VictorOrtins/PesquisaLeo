import cv2
import numpy as np
import os
import skimage as sck

from enum import Enum
from matplotlib import pyplot as plt



class formatoMascara(Enum):
    ELIPSE = "elipse"
    RETANGULAR = "retangulo"

class preProcessamentoMascara(Enum):
    DILATACAO = "dilate"
    MORPH = "morph"
    EROSAO = "erode"

def aplicarTelea(nome: str, imagem, mascara, valor: int, endereco: str):
    telea = cv2.inpaint(imagem, mascara, valor, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(endereco, nome) + '.png', telea)
    
def aplicarNS(nome: str, imagem, mascara, valor: int, endereco: str):
    ns = cv2.inpaint(imagem, mascara, valor, cv2.INPAINT_NS)
    cv2.imwrite(os.path.join(endereco, nome) + '.png', ns)

def aplicarProcessamentoMascara(mascara, m_matriz, n_matriz, preProcessamento: preProcessamentoMascara, formato: formatoMascara, iteracoes=1, tipo=cv2.MORPH_CLOSE):
    if preProcessamento == preProcessamentoMascara.DILATACAO:
        return aplicarDilateMascara(mascara, m_matriz, n_matriz, formato, iteracoes)
    elif preProcessamento == preProcessamentoMascara.EROSAO:
        return aplicarErodeMascara(mascara, m_matriz, n_matriz, formato, iteracoes)
    else:
        return aplicarMorph(mascara, m_matriz, n_matriz, formato, tipo)

def aplicarDilateMascara(mascara, m_matriz, n_matriz, formato: formatoMascara, iteracoes):
    kernel = cria_kernel(m_matriz, n_matriz, formato)
    nova_mascara = cv2.dilate(mascara, kernel, iterations = iteracoes)
    return nova_mascara

def aplicarErodeMascara(mascara, m_matriz, n_matriz, formato: formatoMascara, iteracoes):
    kernel = cria_kernel(m_matriz, n_matriz, formato)
    nova_mascara = cv2.erode(mascara, kernel, iterations = iteracoes)
    return nova_mascara

def aplicarMorph(mascara, m_matriz, n_matriz, formato: formatoMascara, tipo):
    kernel = cria_kernel(m_matriz, n_matriz, formato)
    nova_mascara = cv2.morphologyEx(mascara, tipo, kernel)
    return nova_mascara

def cria_kernel(m_matriz, n_matriz, formato: formatoMascara):
    if formato == formatoMascara.ELIPSE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m_matriz, n_matriz))
    elif formato == formatoMascara.RETANGULAR:
        kernel = np.ones( (m_matriz, n_matriz), np.uint8)

    return kernel

def criaPasta(nome_pasta):
    try:
        os.mkdir(nome_pasta)
    except FileExistsError:
        pass

def mostraImagemOpenCV(nome, imagem):
    cv2.imshow(nome, imagem)
    cv2.waitKey(0)

def mostraImagemScikit(imagem):
        sck.io.imshow(imagem)
        
        plt.show()

def escreveEmArquivo(nome_arquivo, num_linha, texto):
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()

    if num_linha < len(linhas):
        linhas[num_linha] = texto + '\n'
    elif num_linha == len(linhas):
        linhas.append(texto + '\n')

    with open(nome_arquivo, 'w') as arquivo:
        arquivo.writelines(linhas)

def removeArquivosPasta(nome_pasta):
    arquivos = os.listdir(nome_pasta)

    for arquivo in arquivos:
        os.remove(os.path.join(nome_pasta, arquivo))