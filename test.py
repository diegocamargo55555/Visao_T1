import cv2


caminho_img1 = r'imgL.png'
caminho_imagem_direita = r'imgR.png'

imagem1 = cv2.imread(caminho_img1)
imagem2 = cv2.imread(caminho_imagem_direita)

if imagem1 is None or imagem2 is None:
    print("erro ao load imagens")
else:
    print("Imagens carregadas")
    cv2.imshow("Imagem Esquerda", imagem1)
    cv2.imshow("Imagem Direita", imagem2)
    cv2.waitKey(2000) # Aguarda 2 segundos
    cv2.destroyAllWindows()


cv2.destroyAllWindows()