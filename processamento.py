import cv2
import numpy as np

def configurar_algoritmos(tipo_detector, tipo_correspondente):
    if tipo_detector == 'SIFT':
        detector = cv2.SIFT_create()
        parametros_index = dict(algorithm=1, trees=5)
        norma_bf = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        parametros_index = dict(
            algorithm=6, 
            table_number=6, 
            key_size=12, 
            multi_probe_level=1
        )
        norma_bf = cv2.NORM_HAMMING

    if tipo_correspondente == 'BF':
        matcher = cv2.BFMatcher(norma_bf)
    else:
        matcher = cv2.FlannBasedMatcher(parametros_index, dict(checks=50))

    return detector, matcher

def gerar_panorama(imagem_esquerda, imagem_direita, tipo_detector='SIFT', tipo_matcher='FLANN'):
    detector, matcher = configurar_algoritmos(tipo_detector, tipo_matcher)

    kp_esq, des_esq = detector.detectAndCompute(imagem_esquerda, None)
    kp_dir, des_dir = detector.detectAndCompute(imagem_direita, None)

    try:
        correspondencias = matcher.knnMatch(des_dir, des_esq, k=2)
        boas_correspondencias = [
            m[0] for m in correspondencias 
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance
        ]
    except Exception:
        return None
    
    if len(boas_correspondencias) < 4:
        return None

    pts_origem = np.float32([kp_dir[m.queryIdx].pt for m in boas_correspondencias]).reshape(-1, 1, 2)
    pts_destino = np.float32([kp_esq[m.trainIdx].pt for m in boas_correspondencias]).reshape(-1, 1, 2)
    
    matriz_h, _ = cv2.findHomography(pts_origem, pts_destino, cv2.RANSAC, 5.0)
    if matriz_h is None:
        return None

    h_esq, w_esq = imagem_esquerda.shape[:2]
    h_dir, w_dir = imagem_direita.shape[:2]

    cantos_dir = np.float32([[0, 0], [0, h_dir], [w_dir, h_dir], [w_dir, 0]]).reshape(-1, 1, 2)
    cantos_transformados = cv2.perspectiveTransform(cantos_dir, matriz_h)
    
    todos_pontos = np.concatenate(
        (cantos_transformados, np.float32([[0, 0], [0, h_esq], [w_esq, h_esq], [w_esq, 0]]).reshape(-1, 1, 2)), 
        axis=0
    )
    
    x_min, y_min = np.int32(todos_pontos.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(todos_pontos.max(axis=0).ravel() + 0.5)

    matriz_translacao = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    resultado = cv2.warpPerspective(
        imagem_direita, 
        matriz_translacao.dot(matriz_h), 
        (x_max - x_min, y_max - y_min)
    )
    
    resultado[-y_min : h_esq - y_min, -x_min : w_esq - x_min] = imagem_esquerda
    
    return resultado
