# Features de vibração para rolamentos

Extração de **75 features** de uma janela de sinal de vibração, nos domínios do **tempo**, da **frequência** e
**tempo-frequência**. É o mesmo cálculo usado no projeto de estimativa de vida útil remanescente (RUL) de rolamentos
(PRONOSTIA / IEEE PHM 2012), separado aqui para ser reutilizado.

## Instalação

```bash
pip install -r requirements.txt
```

Dependências: `numpy`, `scipy`, `PyWavelets` (e `pandas`, só para o exemplo).

## Uso rápido

```python
import numpy as np
from features_vibracao import extrair_todas_features

fs = 25600                       # frequência de amostragem do seu sensor (Hz)
sinal = np.random.randn(2560)    # troque pela sua janela de vibração (vetor 1D)
feats = extrair_todas_features(sinal, fs)   # dict com 75 features
print(feats["RMS"], feats["Curtose"])
```

Para testar sem dados: `python exemplo_uso.py`, que compara um sinal sintético saudável com um com defeito.

## Funções

| Função | O que calcula | Qtd. |
|---|---|---|
| `calcular_metricas_tempo(sinal)` | máximo, mínimo, média, pico, pico a pico, desvio, RMS, variância, potência, assimetria, curtose, coeficiente de variação, fatores de crista, folga, forma e impulso | 16 |
| `calcular_metricas_frequencia(sinal, fs)` | pela FFT: picos espectrais, banda 1–5 kHz, "harmônicos", bandas laterais, centroide, variância/assimetria/curtose espectral, THD, razão acima/abaixo de 2 kHz, cepstro | 11 |
| `calcular_metricas_tempo_frequencia(sinal, fs)` | STFT e wavelet DWT db4 (4 níveis → 5 sub-bandas); para cada uma, 8 estatísticas: RMS, energia, entropia, média, variância, desvio, assimetria, curtose | 48 |
| `extrair_todas_features(sinal, fs)` | as três acima juntas | 75 |

A explicação de cada feature está nos comentários de `features_vibracao.py`.

## Cuidados

- Cada chamada processa **uma janela de um canal**. Para dois acelerômetros, chame uma vez para cada canal.
- As faixas de frequência são fixas em Hz (1–5 kHz e corte em 2 kHz). Elas fazem sentido para fs ≥ 10 kHz.
- `Harmonicos` é a soma do 2º e do 3º maiores picos do espectro; **não** procura múltiplos da frequência fundamental.
- `THD` é uma variante própria (razão entre potência fora do pico e amplitude do pico).
- A `Curtose` é a de Pearson (sinal gaussiano = 3).
