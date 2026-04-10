import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from processamento import gerar_panorama

class AppPanorama:
    def __init__(self, janela_principal):
        self.janela = janela_principal
        self.janela.title("Gerador de Panorama")
        self.caminhos_imagens = [None, None]

        tk.Label(
            janela_principal, 
            text="Selecione as Imagens", 
            font=("Arial", 12, "bold")
        ).pack(pady=20)

        tk.Button(
            janela_principal, 
            text="Selecionar Imagem Esquerda", 
            command=lambda: self.selecionar_imagem(0),
            width=30
        ).pack(pady=10)

        tk.Button(
            janela_principal, 
            text="Selecionar Imagem Direita", 
            command=lambda: self.selecionar_imagem(1),
            width=30
        ).pack(pady=10)

        self.label_status = tk.Label(janela_principal, text="Nenhuma imagem selecionada", fg="gray")
        self.label_status.pack(pady=15)

        tk.Button(
            janela_principal, 
            text="GERAR PANORAMA", 
            bg="#2ecc71", 
            fg="white", 
            font=("Arial", 10, "bold"), 
            command=self.executar_processamento,
            width=30
        ).pack(pady=20)

    def selecionar_imagem(self, indice):
        caminho = filedialog.askopenfilename(title="Escolha a imagem")
        if caminho:
            self.caminhos_imagens[indice] = caminho
            nome_esq = os.path.basename(self.caminhos_imagens[0]) if self.caminhos_imagens[0] else "..."
            nome_dir = os.path.basename(self.caminhos_imagens[1]) if self.caminhos_imagens[1] else "..."
            self.label_status.config(text=f"Esq: {nome_esq} | Dir: {nome_dir}", fg="black")

    def preparar_imagem_final(self, lista_resultados, espacamento=50):
        largura_maxima = max(img.shape[1] for img in lista_resultados)
        imagens_ajustadas = []
        for i, img in enumerate(lista_resultados):
            pad_largura = largura_maxima - img.shape[1]
            img_pad = np.pad(img, ((0, 0), (0, pad_largura), (0, 0)), mode='constant')
            imagens_ajustadas.append(img_pad)
            
            # Adiciona um espaço preto entre as imagens, exceto após a última
            if i < len(lista_resultados) - 1:
                espaco_vazio = np.zeros((espacamento, largura_maxima, 3), dtype=np.uint8)
                imagens_ajustadas.append(espaco_vazio)
                
        return np.vstack(imagens_ajustadas)

    def mostrar_resultado(self, imagem_bgr, tempo_total, resumo_tempos):
        janela_res = tk.Toplevel(self.janela)
        janela_res.title(f"Panorama Final - {tempo_total:.4f}s")

        frame_texto = tk.Frame(janela_res)
        frame_texto.pack(pady=10)

        tk.Label(
            frame_texto, 
            text=f"Tempo total de execução: {tempo_total:.4f}s", 
            font=("Arial", 11, "bold")
        ).pack()

        tk.Label(
            frame_texto, 
            text=f"Tempos por combinação:\n{resumo_tempos}", 
            font=("Arial", 9), 
            fg="#555",
            justify="left"
        ).pack(pady=5)

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        imagem_pil = Image.fromarray(imagem_rgb)
        imagem_pil.thumbnail((1024, 768))
        
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        label_imagem = tk.Label(janela_res, image=imagem_tk)
        label_imagem.image = imagem_tk
        label_imagem.pack(padx=20, pady=20)

    def executar_processamento(self):
        if not all(self.caminhos_imagens):
            return messagebox.showerror("Erro", "Selecione ambas as imagens antes de continuar!")

        inicio_cronometro = time.time()
        img_esq = cv2.imread(self.caminhos_imagens[0])
        img_dir = cv2.imread(self.caminhos_imagens[1])

        combinacoes = [
            ('ORB', 'BF'), ('ORB', 'FLANN'), 
            ('SIFT', 'BF'), ('SIFT', 'FLANN')
        ]
        
        resultados_validos = []
        tempos_individuais = []
        
        for detector, matcher in combinacoes:
            print(f"Tentando: {detector} + {matcher}...")
            inicio_parcial = time.time()
            
            res = gerar_panorama(img_esq, img_dir, detector, matcher)
            
            tempo_parcial = time.time() - inicio_parcial
            
            if res is not None:
                texto = f"{detector} + {matcher} ({tempo_parcial:.4f}s)"
                cv2.putText(res, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                resultados_validos.append(res)
                tempos_individuais.append((f"{detector} + {matcher}", tempo_parcial))
                print(f" -> ok Tempo: {tempo_parcial:.4f}s")
            else:
                print(f" -> erro.")

        if not resultados_validos:
            return messagebox.showwarning("Não foi possível unir as imagens com nenhum dos algoritmos.")

        tempo_total = time.time() - inicio_cronometro
        
        imagem_final = self.preparar_imagem_final(resultados_validos)
        cv2.imwrite("resultado_panoramico.png", imagem_final)

        resumo_tempos = "\n".join([f"{alg}: {t:.4f}s" for alg, t in tempos_individuais])
        self.mostrar_resultado(imagem_final, tempo_total, resumo_tempos)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("350x300")
    app = AppPanorama(root)
    root.mainloop()
