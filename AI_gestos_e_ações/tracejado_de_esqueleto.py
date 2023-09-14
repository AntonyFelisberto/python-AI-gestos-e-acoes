import cv2
import matplotlib.pyplot as plt
import zipfile
import numpy as np

arquivo_proto = "arquivos\\pose\\body\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "arquivos\\pose\\body\\mpi\\pose_iter_160000.caffemodel"

image = cv2.imread("arquivos\\imagens\\body\\single\\single_3.jpg")

numeros_pontos = 15
pares_pontos = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],
                 [14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]

cor_ponto, cor_linha = (255,128,0),(7,62,248) #cor formato BGR

imagem_largura = image.shape[1]
imagem_altura = image.shape[0]

imagem_copia = np.copy(image)

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto,arquivo_pesos)

altura_entrada = 368
largura_entrada = int((altura_entrada/imagem_altura)*imagem_largura)

blob_entrada = cv2.dnn.blobFromImage(image=image, 
                                     scalefactor=1.0/255,size=(largura_entrada,altura_entrada),
                                     mean=(0,0,0),
                                     swapRB= False,
                                     crop=False
                                    )

modelo.setInput(blob_entrada)
saida = modelo.forward()

altura = saida.shape[2]
largura = saida.shape[3]

pontos = []
limite = 0.1 #quanto maior o limite mais rigoroso vai ser a detecção, 0.6 por exemplo não é tão rigoroso

for i in range(numeros_pontos):
    mapa_confianca = saida[0,i,:,:] #: todos os pontos horizontal : todos os pontos vertical
    _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)
    print(confianca)
    print(ponto)
    x = (imagem_largura * ponto[0]) / largura #redimensionamento para o desenho 
    y = (imagem_altura * ponto[1]) / altura #redimensionamento para o desenho 

    if confianca > limite:
        cv2.circle(imagem_copia,(int(x),int(y)),8,cor_ponto,thickness=-1,lineType=cv2.FILLED)
        cv2.putText(imagem_copia,f"{i}",(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,lineType=cv2.LINE_AA)
        pontos.append((int(x),int(y)))
    else:
        pontos.append(None)

tamanho = cv2.resize(image,(imagem_largura,imagem_altura))
mapa_suave = cv2.GaussianBlur(tamanho,(3,3),0,0)
mascara_mapa = np.uint8(mapa_suave > limite)

for par in pares_pontos:
    parte_a = par[0]
    parte_b = par[1]

    if pontos[parte_a] and pontos[parte_b]:
        cv2.line(image,pontos[parte_a],pontos[parte_b],cor_linha,3)
        cv2.circle(image,pontos[parte_a],8,cor_ponto,thickness=-1,lineType=cv2.LINE_AA)

        cv2.line(mascara_mapa,pontos[parte_a],pontos[parte_b],cor_linha,3)
        cv2.circle(mascara_mapa,pontos[parte_a],8,cor_ponto,thickness=-1,lineType=cv2.LINE_AA)

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(imagem_copia,cv2.COLOR_BGR2RGB))
plt.show()

plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()


plt.figure(figsize=[14,10])
plt.imshow(cv2.cvtColor(mascara_mapa,cv2.COLOR_BGR2RGB))
plt.show()