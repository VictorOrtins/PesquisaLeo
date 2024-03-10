from auxiliar import *
from inpaint import *
from contorno import *

import os



passo_a_passo(100, 'enhanceds100/fingerprints', 'enhanceds100/groundtruths', 'fingerprint_segmentation', 2, 'dilatacao', 'inpainting', 'final', 0.99)
# limiarContorno('1_R_8_4.png', 138, f'./inpaint/ns', './limiar_contorno')
# dilatacaoContorno('4_L_7_1.png', 125, 10, 10, f'./inpaint/ns', './', 'enhancements100/digitais', 'porcentagem.txt')
# otsuContornoBase('./inpaint/ns', './otsu')
# cannyContorno('1_L_1_1.png', 'inpaint/ns', './')
# cannyContornoBase('inpaint/ns', 'canny')
# sobelContornoBase('inpaint/ns', 'sobelPreenchido', 5, 15)
# preProcessamentoScikit('enhancements100/digitais', 'mascara_teste')
# rodarBaseScikit('enhancements100/latentes/', 'mascara_teste', 'inpaint_teste')
# media_ponderada('enhancements100/digitais','inpaint_teste', 'final_teste', 0.99)
# convexHullContornoBase('inpaint/ns', 'convexHull', 10, 30)
# passo_a_passo_segmentation_ns(100, 'enhancements100/latentes', 'enhancements100/digitais', 'fingerprint_segmentation', 'final_segmentation', 0.99)





