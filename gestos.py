import cv2
import numpy as np
import time

try:
    import pyautogui
except ImportError:
    print("Aviso: pyautogui não encontrado. Comandos de teclado desativados.")
    pyautogui = None

# Parâmetros para Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parâmetros para detecção de pontos (Shi-Tomasi)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Câmera não detectada.")
        return

    # Variáveis de controle
    old_gray = None
    p0 = None
    cooldown = 0
    COOLDOWN_TIME = 1.0  # segundos entre gestos
    LIMIAR_MOVIMENTO = 40  # pixels de deslocamento médio para disparar

    print("--- Controle de Slides por Gesto ---")
    print("1. Coloque a mão na caixa azul para iniciar o rastreamento.")
    print("2. Mova a mão rapidamente para os lados.")
    print("3. Pressione 'q' para sair ou 'r' para resetar pontos.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Espelhar para ficar intuitivo
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Desenhar ROI (Região de Interesse) central
        roi_x1, roi_y1 = w//4, h//4
        roi_x2, roi_y2 = 3*w//4, 3*h//4
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        if p0 is not None:
            # Calcular Fluxo Óptico (Lucas-Kanade)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                # Filtrar pontos bons
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Calcular deslocamento médio horizontal (x)
                if len(good_new) > 0:
                    dx = np.mean(good_new[:, 0] - good_old[:, 0])
                    
                    # Mostrar direção na tela
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                        cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

                    # Lógica de Gesto com Cooldown
                    if time.time() - cooldown > COOLDOWN_TIME:
                        if dx > LIMIAR_MOVIMENTO:
                            print(">>> GESTO: DIREITA (Próximo Slide)")
                            if pyautogui: pyautogui.press('right')
                            cooldown = time.time()
                        elif dx < -LIMIAR_MOVIMENTO:
                            print("<<< GESTO: ESQUERDA (Slide Anterior)")
                            if pyautogui: pyautogui.press('left')
                            cooldown = time.time()

                p0 = good_new.reshape(-1, 1, 2)
            
            # Resetar pontos se restarem poucos
            if p1 is None or len(good_new) < 10:
                p0 = None
        
        # UI
        status = "AGUARDANDO MAO" if p0 is None else "RASTREANDO"
        cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Controle de Slides', frame)

        key = cv2.waitKey(30) & 0xff
        if key == ord('q'): break
        elif key == ord('r'): p0 = None # Reset manual
        
        # Inicializar pontos se estiver na ROI e sem pontos
        if p0 is None:
            # Pegar pontos apenas dentro da ROI
            mask = np.zeros_like(frame_gray)
            mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            old_gray = frame_gray.copy()
        else:
            old_gray = frame_gray.copy()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
