#ARQUIVOS PODEM SER ENCONTRADOS NO LINK: https://replit.com/@antonysantos3/Acoes-e-gestos#main.py

import cv2
import matplotlib.pyplot as plt
import zipfile

"""
#DESCONPACTAR ARQUIVOS
pose_path = "arquivos/pose.zip"
zip_object = zipfile.ZipFile(file=pose_path,mode="r")
zip_object.extractall("arquivos/")

pose_path = "arquivos/imagens.zip"
zip_object = zipfile.ZipFile(file=pose_path,mode="r")
zip_object.extractall("arquivos/")
"""

arquivo_proto = "arquivos\\pose\\body\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "arquivos\\pose\\body\\mpi\\pose_iter_160000.caffemodel"

image = cv2.imread("arquivos\\imagens\\body\\multiple\\multiple_1.jpeg")
print(image.shape)

imagem_largura = image.shape[1]
imagem_altura = image.shape[0]
print(imagem_largura,imagem_altura)

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto,arquivo_pesos)

#diminuindo imagem para processamento mais rapido
altura_entrada = 368
largura_entrada = int((altura_entrada/imagem_altura)*imagem_largura) #deixando largura proporcional a altura

blob_entrada = cv2.dnn.blobFromImage(image=image, 
                                     scalefactor=1.0/255,size=(largura_entrada,altura_entrada),
                                     mean=(0,0,0),  #Valores de rgb imagem
                                     swapRB= False,
                                     crop=False #corta imagem
                                    )

modelo.setInput(blob_entrada)
saida = modelo.forward()
print(saida.shape)
print(saida[0].shape)
print(saida[0][1])

ponto = 4   #cada ponto do corpo tem um valor, ai tem que olha o modelo que esta usando para ver o numero do ponto em especifico, no coco são 18 pontos
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")

ponto = 15 
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")

#ponto de afinidade, roxo não tem tanta certeza e amarelo ja tem mais certeza do ponto de conexão, os pontos de afinidade vão de 0 a 43
ponto = 16
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")

ponto = 17
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")

ponto = 18
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")

ponto = 43
mapa_confianca = saida[0, ponto, :, :]
mapa_confianca = cv2.resize(mapa_confianca,(imagem_altura,imagem_largura))

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
plt.imshow(mapa_confianca,alpha=0.6)
plt.axes("off")