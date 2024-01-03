from auxiliar import *


def limiarContorno(nome_imagem, thresh, caminho_inpaint, caminho_final):
    criaPasta(caminho_final)

    inpaint_imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem)

    _, imagem_limiarizada = cv2.threshold(inpaint_imagem, thresh, 255, cv2.THRESH_BINARY) 

    cv2.imwrite(caminho_final + '/' + nome_imagem, imagem_limiarizada)

def dilatacaoContorno(nome_imagem, thresh, dilateRuido, dilateContorno, caminho_inpaint, caminho_final, base_digitais='enhancements100/digitais', nome_arquivo='porcentagem.txt'):
    criaPasta(caminho_final)

    inpaint_imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem)

    _, imagem_limiarizada = cv2.threshold(inpaint_imagem, thresh, 255, cv2.THRESH_BINARY)

    imagem_dilatada = aplicarDilateMascara(imagem_limiarizada, dilateRuido, dilateRuido, formatoMascara.ELIPSE, 1)

    imagem_dilatada = 255 - imagem_dilatada

    imagem_dilatada = aplicarDilateMascara(imagem_dilatada, dilateContorno, dilateContorno, formatoMascara.ELIPSE, 1)

    imagem_dilatada = 255 - imagem_dilatada

    digitais = os.listdir(base_digitais)
    digitais.sort()
    index = digitais.index(nome_imagem)

    texto = str(thresh) + ',' + str(dilateRuido) + ',' + str(dilateContorno)
    escreveEmArquivo(caminho_final + '/' + nome_arquivo, index, texto)


def otsuContorno(nome_imagem, caminho_inpaint, caminho_final):
    criaPasta(caminho_final)

    inpaint_imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem, cv2.IMREAD_GRAYSCALE)

    _, imagem_limiarizada = cv2.threshold(inpaint_imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(caminho_final + '/' + nome_imagem, imagem_limiarizada)


def otsuContornoBase(caminho_inpaint, caminho_final):
    arquivos = os.listdir(caminho_inpaint)

    for arquivo in arquivos:
        otsuContorno(arquivo, caminho_inpaint, caminho_final)

def cannyContorno(nome_imagem, caminho_inpaint, caminho_final, canny1, canny2):
    imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem, cv2.IMREAD_GRAYSCALE)

    canny = cv2.Canny(imagem, canny1, canny2)

    contornos, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagem_contorno = np.zeros( (imagem.shape[0], imagem.shape[1], 3), dtype=np.uint8)
    imagem_contorno = 255 - imagem_contorno

    cv2.drawContours(imagem_contorno, contornos, -1, (0, 0, 0), 2)

    cv2.imwrite(caminho_final + '/' + nome_imagem, imagem_contorno)
    
def cannyContornoBase(caminho_inpaint, caminho_final):
    arquivos = os.listdir(caminho_inpaint)

    for arquivo in arquivos:
        cannyContorno(arquivo, caminho_inpaint, caminho_final, 10, 30)

def sobelContorno(nome_imagem, caminho_inpaint, caminho_final, tam_kernel_sobel, tam_kernel_gauss):
    imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem, cv2.IMREAD_GRAYSCALE)

    imagem = cv2.GaussianBlur(imagem, (tam_kernel_gauss, tam_kernel_gauss), 0)

    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=tam_kernel_sobel)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=tam_kernel_sobel)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    bordas_combinadas = cv2.bitwise_or(sobel_x, sobel_y)

    _, imagem_limiarizada = cv2.threshold(bordas_combinadas, 80, 255, cv2.THRESH_BINARY)

    seed = (int(imagem_limiarizada.shape[0]/2), int(imagem_limiarizada.shape[1]/2))

    cv2.floodFill(imagem_limiarizada, None, seed, (255,255,255))

    cv2.imwrite(caminho_final + '/' + nome_imagem, imagem_limiarizada)

def sobelContornoBase(caminho_inpaint, caminho_final, tam_kernel_sobel, tam_kernel_gauss):
    arquivos = os.listdir(caminho_inpaint)

    for arquivo in arquivos:
        sobelContorno(arquivo, caminho_inpaint, caminho_final, tam_kernel_sobel, tam_kernel_gauss)

def convexHullContorno(nome_imagem, caminho_inpaint, caminho_final, canny1, canny2):
    imagem = cv2.imread(caminho_inpaint + '/' + nome_imagem)

    canny = cv2.Canny(imagem, canny1, canny2)

    contornos, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for c in contornos:
        hull = cv2.convexHull(c)
        hull_list.append(hull)

    imgBranca = np.ones( (imagem.shape[0], imagem.shape[1]), dtype='uint8')

    for c in contornos:
        cv2.drawContours(imgBranca, contornos, -1, (255,255,255), thickness=-1)
        cv2.drawContours(imgBranca, hull_list,-1, (255,255,255), thickness=-1)

    imgBranca = 255 - imgBranca

    cv2.imwrite(caminho_final + '/' + nome_imagem, imgBranca)

def convexHullContornoBase(caminho_inpaint, caminho_final, canny1, canny2):
    arquivos = os.listdir(caminho_inpaint)

    for arquivo in arquivos:
        convexHullContorno(arquivo, caminho_inpaint, caminho_final, canny1, canny2)
        print(arquivo)


    

