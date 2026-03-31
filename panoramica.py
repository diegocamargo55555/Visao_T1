import cv2
import numpy as np
import time

def criar_panoramica(img_esq, img_dir, detector_tipo, matcher_tipo):
    tempo_inicio = time.time()

    cinza_esq = cv2.cvtColor(img_esq, cv2.COLOR_BGR2GRAY)
    cinza_dir = cv2.cvtColor(img_dir, cv2.COLOR_BGR2GRAY)

    if detector_tipo == 'sift':
        detector = cv2.SIFT_create()
    elif detector_tipo == 'orb':
        detector = cv2.ORB_create(nfeatures=2000) 
    else:
        return None, 0

    kps_esq, des_esq = detector.detectAndCompute(cinza_esq, None)
    kps_dir, des_dir = detector.detectAndCompute(cinza_dir, None)

    if des_esq is None or des_dir is None:
        print("Erro: Não foram encontradas características suficientes nas imagens.")
        return None, 0

    if matcher_tipo == 'bf':
        norma = cv2.NORM_L2 if detector_tipo == 'sift' else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norma, crossCheck=False)
        
    elif matcher_tipo == 'flann':
        if detector_tipo == 'sift':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = matcher.knnMatch(des_dir, des_esq, k=2)
    except Exception as e:
        print(f"Erro na correspondência com o algoritmo selecionado: {e}")
        return None, 0

    bons_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                bons_matches.append(m)

    if len(bons_matches) > 10: 
        pontos_dir = np.float32([kps_dir[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
        pontos_esq = np.float32([kps_esq[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

        H, mascara = cv2.findHomography(pontos_dir, pontos_esq, cv2.RANSAC, 5.0)

        if H is None:
            print("Erro: Não foi possível calcular a matriz de Homografia.")
            return None, 0

        largura = img_esq.shape[1] + img_dir.shape[1]
        altura = img_esq.shape[0]

        img_resultante = cv2.warpPerspective(img_dir, H, (largura, altura))
        img_resultante[0:img_esq.shape[0], 0:img_esq.shape[1]] = img_esq

        gray_res = cv2.cvtColor(img_resultante, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            x, y, w, h = cv2.boundingRect(contornos[0])
            img_panoramica = img_resultante[y:y+h, x:x+w]
        else:
            img_panoramica = img_resultante

        tempo_fim = time.time()
        tempo_total = tempo_fim - tempo_inicio

        return img_panoramica, tempo_total

    else:
        print(f"Erro: Pontos correspondentes insuficientes ({len(bons_matches)} encontrados).")
        return None, 0