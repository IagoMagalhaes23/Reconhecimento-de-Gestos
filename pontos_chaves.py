import cv2
import matplotlib.pyplot as plt
import zipfile
import numpy as np

print(cv2.__version__)

arquivo_proto = "C:/Users/Iago/Desktop/Reconhecimento de gestos/pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "C:/Users/Iago/Desktop/Reconhecimento de gestos/pose/body/mpi/pose_iter_160000.caffemodel"

imagem = cv2.imread("img.jpeg")

numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],[1,14],
               [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

cor_ponto, cor_linha = (255, 128, 0), (7, 62, 248)

imagem_copia = np.copy(imagem)

imagem_largura = imagem.shape[1]
imagem_altura = imagem.shape[0]

imagem_largura, imagem_altura

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

altura_entrada = 368
largura_entrada = int((altura_entrada / imagem_altura) * imagem_largura)

blob_entrada = cv2.dnn.blobFromImage(imagem, 1.0 / 255, 
                                    (largura_entrada, altura_entrada), 
                                    (0, 0, 0), swapRB = False, crop = False)

modelo.setInput(blob_entrada)
saida = modelo.forward()

altura = saida.shape[2]
largura = saida.shape[3]

altura, largura

pontos = []
limite = 0.1
for i in range(numero_pontos):
  mapa_confianca = saida[0, i, :, :]
  _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)
  #print(confianca)
  #print(ponto)
  
  x = (imagem_largura * ponto[0]) / largura
  y = (imagem_altura * ponto[1] / altura)
  
  if confianca > limite:
    cv2.circle(imagem_copia, (int(x), int(y)), 8, cor_ponto, thickness = -1, 
               lineType=cv2.FILLED)
    cv2.putText(imagem_copia, "{}".format(i), (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, 
                lineType=cv2.LINE_AA)
    pontos.append((int(x), int(y)))
  else:
    pontos.append(None)

tamanho = cv2.resize(imagem, (imagem_largura, imagem_altura))
mapa_suave = cv2.GaussianBlur(tamanho, (3,3), 0, 0)
mascara_mapa = np.uint8(mapa_suave > limite)

for par in pares_pontos:
  #print(par)
  parteA = par[0]
  parteB = par[1]
  
  if pontos[parteA] and pontos[parteB]:
    cv2.line(imagem, pontos[parteA], pontos[parteB], cor_linha, 3)
    cv2.circle(imagem, pontos[parteA], 8, cor_ponto, thickness = -1,
              lineType = cv2.LINE_AA)
    
    cv2.line(mascara_mapa, pontos[parteA], pontos[parteB], cor_linha, 3)
    cv2.circle(mascara_mapa, pontos[parteA], 8, cor_ponto, thickness = -1,
              lineType = cv2.LINE_AA)

cv2.imwrite("pontos_chave.jpg", cv2.cvtColor(imagem_copia, cv2.COLOR_BGR2RGB))

cv2.imwrite("pontos.jpg", cv2.cvtColor(mascara_mapa, cv2.COLOR_BGR2RGB))