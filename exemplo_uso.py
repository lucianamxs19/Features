"""
exemplo_uso.py
==============
Exemplo que roda sem nenhum dado: cria dois sinais sintéticos de vibração (rolamento saudável e com defeito),
extrai as 75 features de cada um e mostra a diferença nas principais.

Rodar:   python exemplo_uso.py
"""
import numpy as np
import pandas as pd

from features_vibracao import extrair_todas_features

FS = 25600            # frequência de amostragem (Hz)
N = 2560              # 0,1 s de sinal


def sinal_sintetico(defeito, seed=0):
    """Ruído + rotação do eixo; com defeito, impactos periódicos excitando uma ressonância de 3 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = 0.05 * np.sin(2 * np.pi * 30 * t) + 0.1 * rng.standard_normal(N)       # eixo a 30 Hz + ruído
    if defeito:
        for t0 in np.arange(0.002, t[-1], 1 / 168):                             # um impacto a cada 1/168 s
            k = t >= t0
            tau = t[k] - t0
            x[k] += 1.0 * np.exp(-600 * tau) * np.sin(2 * np.pi * 3000 * tau)
    return x


if __name__ == "__main__":
    tabela = pd.DataFrame([extrair_todas_features(sinal_sintetico(False), FS),
                           extrair_todas_features(sinal_sintetico(True), FS)], index=["saudável", "com defeito"])
    print(f"{tabela.shape[1]} features por janela\n")
    principais = ["RMS", "Pico_a_Pico", "Curtose", "Fator_Crista", "Centroide_Espectral", "Energy_DWT_L3"]
    print(tabela[principais].T.round(3).to_string())
    tabela.T.to_csv("features_exemplo.csv", sep=";", decimal=",")
    print("\nTabela completa salva em features_exemplo.csv")
