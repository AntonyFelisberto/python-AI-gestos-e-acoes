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

arquivo_proto = "AI_gestos_e_acoes\\arquivos\\pose\\body\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "AI_gestos_e_acoes\\arquivos\\pose\\body\\mpi\\pose_iter_160000.caffemodel"

numeros_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

cor_ponto_A, cor_ponto_B, cor_linha =  (14, 201,255), (255,0,128), (192,192,192)
cor_txt_ponto, cor_txt_inicial, cor_txt_andamento =  (10, 216,245), (255,0,128), (192,192,192)

tamanho_fonte,tamanho_linha,tamanho_circulo,espessura = 0.8,2,8,5
fonte = cv2.FONT_HERSHEY_SIMPLEX
valida_pernas_juntas, valida_pernas_afastadas = 0,0
valida_bracos_abaixo, valida_bracos_acima = 0,0

entrada_largura = 256
entrada_altura = 256

video = "AI_gestos_e_acoes\\videos\\polichinelos.mp4"
captura = cv2.VideoCapture(video)
conectado, frame = captura.read()

resultado = "polichinelo.avi"
grava_video = cv2.VideoWriter(resultado,cv2.VideoWriter_fourcc(*"XVID"),10,(frame.shape[1],frame.shape[0]))

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto,arquivo_pesos)

limite = 0.1
while cv2.waitKey(1) < 0:
    t = time.time()
    conectado,video = captura.read()
    video_copia = np.copy(video)
    if not conectado:
        cv2.waitKey()
        break

    video_largura = video.shape[1]
    video_altura = video.shape[0]

    tamanho = cv2.resize(video,(video_largura,video_altura))
    mapa_suave = cv2.GaussianBlur(tamanho,(3,3),0,0)
    fundo = np.uint8(mapa_suave > limite)

    blob_entrada = cv2.dnn.blobFromImage(video, 
                                        scalefactor=1.0/255,size=(entrada_largura,entrada_altura),
                                        mean=(0,0,0),
                                        swapRB= False,
                                        crop=False
                                        )
    modelo.setInput(blob_entrada)
    saida = modelo.forward()
    altura = saida.shape[2]
    largura = saida.shape[3]
    pontos = []

    for i in range(numeros_pontos):
        mapa_confianca = saida[0,i,:,:] #: todos os pontos horizontal : todos os pontos vertical
        _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)
        print(confianca)
        print(ponto)
        x = (video_largura * ponto[0]) / largura #redimensionamento para o desenho 
        y = (video_altura * ponto[1]) / altura #redimensionamento para o desenho 

        if confianca > limite:
            cv2.circle(video_copia,(int(x),int(y)),4,cor_ponto_B,thickness=tamanho_circulo,lineType=cv2.FILLED)
            cv2.putText(video_copia,f"{i}",(int(x),int(y)),fonte,tamanho_fonte,cor_txt_ponto,3,lineType=cv2.LINE_AA)
            cv2.putText(fundo,f" ",(int(x),int(y)),fonte,tamanho_fonte,cor_txt_ponto,3,lineType=cv2.LINE_AA)
            pontos.append((int(x),int(y)))
        else:
            pontos.append((0, 0))

    for par in pares_pontos:
        parte_a = par[0]
        parte_b = par[1]

        if pontos[parte_a] and pontos[parte_b]:
            cv2.line(video,pontos[parte_a],pontos[parte_b],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
            cv2.line(video_copia,pontos[parte_a],pontos[parte_b],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
            cv2.line(fundo,pontos[parte_a],pontos[parte_b],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
            cv2.circle(video,pontos[parte_a],4,cor_ponto_A,thickness=espessura,lineType=cv2.FILLED)
            cv2.circle(video,pontos[parte_b],4,cor_ponto_A,thickness=espessura,lineType=cv2.FILLED)

            cv2.circle(fundo,pontos[parte_a],4,cor_ponto_A,thickness=espessura,lineType=cv2.FILLED)
            cv2.circle(fundo,pontos[parte_b],4,cor_ponto_A,thickness=espessura,lineType=cv2.FILLED)

    if extrator.verificar_bracos_ABAIXO(pontos[0:8]) == True:
        valida_bracos_abaixo = 0.25
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Bracos: Posicao inicial",(50,50),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)
    elif extrator.verificar_bracos_ACIMA(pontos[0:8]) == True:
        valida_bracos_acima = 0.25
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Bracos: Posicao final",(50,50),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)
    else:
        valida_bracos_acima = 0
        valida_bracos_abaixo = 0
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Bracos: em andamento",(50,50),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)

    if extrator.verificar_pernas_AFASTADAS(pontos[8:14]) == True:
        valida_pernas_afastadas = 0.5
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Pernas: Posicao final",(50,70),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)
    elif extrator.verificar_pernas_JUNTAS(pontos[8:14]) == True:
        valida_pernas_juntas = 0.25
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Pernas: Posicao inicial",(50,70),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)
    else:
        valida_pernas_afastadas = 0
        valida_pernas_juntas = 0
        cv2.line(video_copia,pontos[0],pontos[1],cor_linha,tamanho_linha,lineType=cv2.LINE_AA)
        cv2.putText(video_copia,f"Pernas: em andamento",(50,70),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)

    if valida_bracos_acima != 0 and valida_pernas_afastadas != 0:
        cv2.putText(video_copia,f"Polichinelo valido {valida_bracos_acima + valida_pernas_afastadas}",(50,200),fonte,tamanho_fonte,cor_txt_inicial,0,lineType=cv2.LINE_AA)

    cv2.putText(video_copia,f"tempo por frame {time.time() - t} valido {valida_bracos_acima + valida_pernas_afastadas}",(50,20),fonte,tamanho_fonte,(40,40,40),0,lineType=cv2.LINE_AA)

    grava_video.write(video_copia)
grava_video.release()