import cv2
from panoramica import criar_panoramica

def menu_interativo():
    imgL = 'imgL.png'
    imgR = 'imgR.png'

    imgL = cv2.imread(imgL)
    imgR = cv2.imread(imgR)

    if imgL is None or imgR is None:
        print(f"Erro ao carregar as imagens")
        return

    while True:
        print("\n" + "="*50)
        print(" MENU DE PANORÂMICA (Interface Interativa) ")
        print("="*50)
        print("Selecione a combinação de algoritmos:")
        print("1. ORB e BF")
        print("2. ORB e FLANN")
        print("3. SIFT e BF")
        print("4. SIFT e FLANN")
        print("0. Sair")
        print("="*50)

        opcao = input("Digite a sua opção: ")

        if opcao == '0':
            print("Encerrando o programa...")
            break
            
        opcoes_map = {
            '1': ('orb', 'bf'),
            '2': ('orb', 'flann'),
            '3': ('sift', 'bf'),
            '4': ('sift', 'flann')
        }

        if opcao not in opcoes_map:
            print("Opção inválida! Tente novamente.")
            continue

        detector_tipo, matcher_tipo = opcoes_map[opcao]

        print(f"\nProcessando panorâmica com {detector_tipo.upper()} + {matcher_tipo.upper()}...")
        
        panoramica, tempo = criar_panoramica(imgL, imgR, detector_tipo, matcher_tipo)

        if panoramica is not None:
            print(f"Sucesso! Tempo de processamento: {tempo:.4f} segundos")
            
            titulo_janela = f"Panoramica: {detector_tipo.upper()} + {matcher_tipo.upper()}"
            
            escala = 0.5
            altura = int(panoramica.shape[0] * escala)
            largura = int(panoramica.shape[1] * escala)
            panoramica_redimensionada = cv2.resize(panoramica, (largura, altura))
            
            cv2.imshow(titulo_janela, panoramica_redimensionada)
            print("Pressione qualquer tecla na janela da imagem para fechar e voltar ao menu.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    menu_interativo()