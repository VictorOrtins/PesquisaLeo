import os
import cv2
from auxiliar import *
import threading
import skimage as sck
import numpy as np    


def preProcessamento(caminho_mascaras, tam_mascara, final_mascara):
    nomes_arquivos = os.listdir(caminho_mascaras) 

    enderecoMascaraTelea = os.path.join(final_mascara, 'telea')
    enderecoMascaraNs = os.path.join(final_mascara, 'ns')

    try:
        os.mkdir(final_mascara)
    except FileExistsError as e:
        pass

    try:
        os.mkdir(enderecoMascaraTelea)
    except FileExistsError as e:
        pass    

    try:
        os.mkdir(enderecoMascaraNs)
    except FileExistsError as e:
        pass

    for c in range(0, len(nomes_arquivos)):
        nomeArquivo: str = nomes_arquivos[c][0:nomes_arquivos[c].find('.')]

        # Mascara usada na imagem
        mascara = cv2.imread(os.path.join(caminho_mascaras, nomes_arquivos[c]))

        mascara = 255 - mascara

        # mascara = aplicarProcessamentoMascara(mascara, tam_mascara, tam_mascara, preProcessamentoMascara.DILATACAO, formatoMascara.ELIPSE, 1, None)

        cv2.imwrite(os.path.join(enderecoMascaraNs, nomeArquivo) + '.png', mascara)

        mascara = aplicarProcessamentoMascara(mascara, tam_mascara, tam_mascara, preProcessamentoMascara.DILATACAO, formatoMascara.ELIPSE, 1, None)

        cv2.imwrite(os.path.join(enderecoMascaraTelea, nomeArquivo) + '.png', mascara)


def rodar_base(valor_inpaint, caminho_mascaras, caminho_digitais, final_inpaint):
    # Obtém os nomes dos arquivos no diretório

    threads: list = []
        
    valor: int = valor_inpaint

    enderecoMascaraTelea = os.path.join(caminho_mascaras, 'telea')
    enderecoMascaraNs = os.path.join(caminho_mascaras, 'ns')

    nomes_arquivos = os.listdir(enderecoMascaraNs)


    salvarTelea = os.path.join(final_inpaint, 'telea')
    salvarNs = os.path.join(final_inpaint, 'ns')

    try:
        os.mkdir(final_inpaint)
    except FileExistsError as e:
        pass
    
    try:
        os.mkdir(salvarTelea)
    except FileExistsError as e:
        pass

    try:
        os.mkdir(salvarNs)
    except FileExistsError as e:
        pass

    for c in range(0, len(nomes_arquivos)):
        nome_arquivo: str = nomes_arquivos[c][0:nomes_arquivos[c].find('.')]

        # Carregar a imagem à qual será aplicada a máscara
        imagem = cv2.imread(os.path.join(caminho_digitais, nomes_arquivos[c]).split('.')[0] + '.bmp')
        
        mascaraTelea = cv2.imread(os.path.join(enderecoMascaraTelea, nome_arquivo) + '.png', cv2.IMREAD_GRAYSCALE)
        mascaraNs = cv2.imread(os.path.join(enderecoMascaraNs, nome_arquivo) + '.png', cv2.IMREAD_GRAYSCALE)

        thread1 = threading.Thread(target=aplicarTelea,args=(nome_arquivo, imagem, mascaraTelea, valor, salvarTelea))
        thread2 = threading.Thread(target=aplicarNS,args=(nome_arquivo, imagem, mascaraNs, valor, salvarNs))

        # Aplicar o inpainting na imagem
        thread1.start()
        thread2.start()

        threads.append(thread1)
        threads.append(thread2)

        # Controlando a quantidade de threads criadas
        if (len(threads) == 10):
            for thread in threads:
                thread.join()
            threads.clear()
        
    if (len(threads) != 0):
        for thread in threads:
            thread.join()

def media_ponderada(caminho_fingerprints, caminho_groundtruths, caminho_segmentation, caminho_inpaint, caminho_final, t):
    nomes_arquivos = os.listdir(caminho_fingerprints)

    endereco_telea = os.path.join(caminho_inpaint, 'telea')
    endereco_ns = os.path.join(caminho_inpaint, 'ns')

    arquivos_telea = os.listdir(endereco_telea)
    arquivos_NS = os.listdir(endereco_ns)
    arquivos_segmentados = os.listdir(caminho_segmentation)

    try:
        os.mkdir(caminho_final)
    except FileExistsError as e:
        pass
    
    try:
        os.mkdir(os.path.join(caminho_final, 'ns'))
    except FileExistsError as e:
        pass

    try:
        os.mkdir(os.path.join(caminho_final, 'telea'))
    except FileExistsError as e:
        pass


    for c in range(0, len(nomes_arquivos)):
        nome_arquivo: str = nomes_arquivos[c][0:nomes_arquivos[c].find('.')]

        imagem_enhanced = cv2.imread(os.path.join(caminho_groundtruths, nomes_arquivos[c].split('.')[0] + '.png'))

        try:
            os.mkdir(os.path.join(caminho_final, 'ns', nome_arquivo))
        except FileExistsError:
            pass

        try:
            os.mkdir(os.path.join(caminho_final, 'telea', nome_arquivo))
        except FileExistsError:
            pass

        for d in range(len(arquivos_NS)):
            segmentado = cv2.imread(os.path.join(caminho_segmentation, arquivos_segmentados[d]))
            segmentado = 255 - segmentado

            enhanced_ponderada = cv2.bitwise_or(imagem_enhanced, segmentado)

            imagem_telea = cv2.imread(os.path.join(endereco_telea, arquivos_telea[d]))
            imagem_ns = cv2.imread(os.path.join(endereco_ns, arquivos_NS[d]))

            ns = cv2.addWeighted(enhanced_ponderada, 1-t, imagem_ns, t, 0)
            telea = cv2.addWeighted(enhanced_ponderada, 1-t, imagem_telea, t, 0)


            cv2.imwrite(os.path.join(caminho_final, 'ns', nome_arquivo, arquivos_NS[d]), ns)
            cv2.imwrite(os.path.join(caminho_final, 'telea', nome_arquivo, arquivos_telea[d]), telea)

def passo_a_passo(valor_inpaint, caminho_fingerprints, caminho_groundtruths, caminho_segmentation, tam_mascara, final_mascara, final_inpaint, caminho_final, t):
    # preProcessamento(caminho_groundtruths, tam_mascara, final_mascara)
    # rodar_base(valor_inpaint, final_mascara, caminho_fingerprints, final_inpaint)
    media_ponderada(caminho_fingerprints, caminho_groundtruths, caminho_segmentation, final_inpaint, caminho_final, t)

def preProcessamentoScikit(caminho_base, finalMascara):
    arquivos = os.listdir(caminho_base)

    for arquivo in arquivos:
        enderecoImagem = caminho_base + '/' + arquivo

        img = sck.io.imread(enderecoImagem)

        img = np.invert(img)

        img = sck.morphology.dilation(img)

        sck.io.imsave(finalMascara + '/' + arquivo, img)

def rodarBaseScikit(caminho_base, caminho_mascara, final_inpaint):
    arquivos = os.listdir(caminho_base)

    for arquivo in arquivos:
        enderecoLatente = caminho_base + '/' + arquivo
        enderecoMascara = caminho_mascara + '/' + arquivo.split('.')[0] + '.png'

        latente = sck.io.imread(enderecoLatente)
        mascara = sck.io.imread(enderecoMascara)

        imgResult = sck.restoration.inpaint_biharmonic(latente, mascara)
        imgResult = sck.img_as_ubyte(imgResult)


        sck.io.imsave(final_inpaint + '/' + arquivo.split('.')[0] + '.png', imgResult)

def mediaPonderadaScikit(caminho_base, enderecoInpaint, enderecoFinal, t):
    nomes_arquivos = os.listdir(caminho_base)

    arquivos_NS = os.listdir(enderecoInpaint)

    print(len(nomes_arquivos))
    print(len(arquivos_NS))


    for c in range(len(nomes_arquivos)):
        nomeArquivo: str = nomes_arquivos[c].split('.')[0]

        imagemEnhanced = cv2.imread(f'./{caminho_base}/' + nomes_arquivos[c])

        try:
            os.mkdir(enderecoFinal + '/' + nomeArquivo.split('.')[0])
        except FileExistsError:
            pass

        for d in range(len(arquivos_NS)):
            imagemNS = cv2.imread(enderecoInpaint + '/' + arquivos_NS[d])

            ns = cv2.addWeighted(imagemEnhanced, 1-t, imagemNS, t, 0)

            cv2.imwrite(f'{enderecoFinal}/{nomeArquivo}/{arquivos_NS[d]}', ns)

def passo_a_passo_scikit(caminho_base, final_mascara, final_inpaint, final, t):
    preProcessamentoScikit(caminho_base, final_mascara)
    rodarBaseScikit(caminho_base, final_mascara, final_inpaint)
    mediaPonderadaScikit(caminho_base, final_inpaint, final, t)

def passo_a_passo_segmentation_ns(valor_inpaint, caminho_latentes, caminho_enhanceds, caminho_segmentado, caminho_final, t):
    latentes = os.listdir(caminho_latentes)
    latentes.sort()
    enhanceds = os.listdir(caminho_enhanceds)
    enhanceds.sort()
    segmentados = os.listdir(caminho_segmentado)
    segmentados.sort()

    caminho_final_fingerprints = os.path.join(caminho_final, 'fingerprints')
    caminho_final_groundtruths = os.path.join(caminho_final, 'groundtruths')

    try:
        os.mkdir(caminho_final)
    except FileExistsError:
        pass

    try:
        os.mkdir(caminho_final_fingerprints)
    except FileExistsError:
        pass

    try:
        os.mkdir(caminho_final_groundtruths)
    except FileExistsError:
        pass

    contador_img = 0

    for i in range(len(latentes)):
        latente = cv2.imread(os.path.join(caminho_latentes, latentes[i]))

        enhanced_inpaint = cv2.imread(os.path.join(caminho_enhanceds, enhanceds[i]), cv2.IMREAD_GRAYSCALE)

        print('inpaint')
        enhanced_inpaint = 255 - enhanced_inpaint

        ns = cv2.inpaint(latente, enhanced_inpaint, valor_inpaint, cv2.INPAINT_NS)

        segmentado = cv2.imread(os.path.join(caminho_segmentado, segmentados[i]))
        segmentado = 255 - segmentado


        for j in range(len(enhanceds)):
            enhanced_ponderada = cv2.imread(os.path.join(caminho_enhanceds, enhanceds[j]))

            enhanced_ponderada = cv2.bitwise_or(enhanced_ponderada, segmentado)
            
            fingerprint = cv2.addWeighted(enhanced_ponderada, 1-t, ns, t, 0)
            
            cv2.imwrite(os.path.join(caminho_final_groundtruths, f'({contador_img}).png'), enhanced_ponderada)
            cv2.imwrite(os.path.join(caminho_final_fingerprints, f'({contador_img}).png'), fingerprint)

            contador_img += 1
















        



    







