import cv2
import numpy as np
import pyautogui

THRESHOLD_MOVIMENTO = 15  
COOLDOWN_FRAMES = 10      
MIN_PONTOS_ATIVOS = 5     

lk_params = dict(winSize=(31, 31), 
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100, 
                      qualityLevel=0.01, 
                      minDistance=5, 
                      blockSize=10)

cap = cv2.VideoCapture(0)
nome_janela = 'Interface Gestual'
cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
cv2.resizeWindow(nome_janela, 1000, 700)

p0 = None
contador_cooldown = 0


while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if contador_cooldown <= 0:
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

    if p0 is not None and 'gray_antiga' in locals():
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_antiga, gray, p0, None, **lk_params)

        if p1 is not None and len(p1[st == 1]) > MIN_PONTOS_ATIVOS:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            dx = good_new[:, 0] - good_old[:, 0]
            media_x = np.mean(dx)

            for pt in good_new:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            if contador_cooldown <= 0:
                if media_x > THRESHOLD_MOVIMENTO:
                    print(">> GESTO: DIREITA")
                    pyautogui.press('right')
                    contador_cooldown = COOLDOWN_FRAMES
                elif media_x < -THRESHOLD_MOVIMENTO:
                    print("<< GESTO: ESQUERDA")
                    pyautogui.press('left')
                    contador_cooldown = COOLDOWN_FRAMES

    gray_antiga = gray.copy()
    if contador_cooldown > 0:
        contador_cooldown -= 1

    status = "PRONTO" if contador_cooldown <= 0 else "AGUARDANDO"
    cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow(nome_janela, frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()