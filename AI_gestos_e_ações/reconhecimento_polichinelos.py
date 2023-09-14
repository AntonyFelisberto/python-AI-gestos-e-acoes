import time
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import arquivos.modulos.extrator_CORPO as extrator

""" descompactar aquivos
pose_path = "pose.zip"
zip_object = zipfile.ZipFile(file=pose_path, mode="r")
zip_object.extractall("./")

imagens_path = "imagens.zip"
zip_object = zipfile.ZipFile(file=imagens_path, mode="r")
zip_object.extractall("./")

modulos_path = "modulos.zip"
zip_object = zipfile.ZipFile(file=modulos_path, mode="r")
zip_object.extractall("./")
zip_object.close()
"""

arquivo_proto = "arquivos\\pose\\body\\body\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "arquivos\\pose\\body\\body\\mpi\\pose_iter_160000.caffemodel"

numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

cor_ponto_A, cor_ponto_B, cor_linha =  (14, 201,255), (255,0,128), (192,192,192)
cor_txt_ponto, cor_txt_inicial, cor_txt_andamento =  (10, 216,245), (255,0,128), (192,192,192)

tamanho_fonte,tamanho_linha,tamanho_circulo,espessura = 9.8,2,8,5
fonte = cv2.FONT_HERSHEY_SIMPLEX
valida_pernas_juntas