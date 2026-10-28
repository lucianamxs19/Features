# Features de vibração para rolamentos

Extração de até **88 features** de uma janela de sinal de vibração. São **75 básicas**, nos domínios do **tempo**, da
**frequência** e **tempo-frequência** (`features_vibracao.py`), e **13 avançadas**: entropias, Hjorth, kurtograma e envelope
nas frequências de defeito (`features_avancadas.py`). É o mesmo banco de features usado no projeto de estimativa de vida
útil remanescente (RUL) de rolamentos (PRONOSTIA / IEEE PHM 2012), separado aqui para ser reutilizado.

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

Banco completo de 88 features. As 4 de envelope precisam da geometria do rolamento e da rotação:

```python
from features_avancadas import BearingGeometry
from features_vibracao import extrair_88_features

geom = BearingGeometry(n_balls=13, ball_diameter_mm=3.5, pitch_diameter_mm=26.0, contact_angle_deg=0.0)
feats = extrair_88_features(sinal, fs, geom=geom, rpm=1800)   # 88 features (sem geom/rpm: 84)
```

Para testar sem dados: `python exemplo_uso.py`, que compara um sinal sintético saudável com um com defeito.

## Funções

| Função | O que calcula | Qtd. |
|---|---|---|
| `calcular_metricas_tempo(sinal)` | máximo, mínimo, média, pico, pico a pico, desvio, RMS, variância, potência, assimetria, curtose, coeficiente de variação, fatores de crista, folga, forma e impulso | 16 |
| `calcular_metricas_frequencia(sinal, fs)` | pela FFT: picos espectrais, banda 1–5 kHz, "harmônicos", bandas laterais, centroide, variância/assimetria/curtose espectral, THD, razão acima/abaixo de 2 kHz, cepstro | 11 |
| `calcular_metricas_tempo_frequencia(sinal, fs)` | STFT e wavelet DWT db4 (4 níveis → 5 sub-bandas); para cada uma, 8 estatísticas: RMS, energia, entropia, média, variância, desvio, assimetria, curtose | 48 |
| `extrair_todas_features(sinal, fs)` | as três acima juntas | 75 |
| `features_avancadas.features_entropia(sinal)` | entropias de permutação, de amostra e de dispersão | 3 |
| `features_avancadas.hjorth_parameters(sinal)` | Hjorth: atividade, mobilidade, complexidade | 3 |
| `features_avancadas.kurtogram_simplificado(sinal, fs)` | curtose máxima do envelope em 8 bandas e a banda ótima | 3 |
| `features_avancadas.features_envelope_rolamento(sinal, fs, geom, rpm)` | envelope (Hilbert) em BPFO, BPFI, BSF, FTF | 4 |
| `extrair_88_features(sinal, fs, geom, rpm)` | as 75 básicas + as 13 avançadas | 88 |

A explicação de cada feature está nos comentários de `features_vibracao.py` e `features_avancadas.py`.

## Cuidados

- Cada chamada processa **uma janela de um canal**. Para dois acelerômetros, chame uma vez para cada canal.
- As faixas de frequência são fixas em Hz (1–5 kHz e corte em 2 kHz). Elas fazem sentido para fs ≥ 10 kHz.
- `Harmonicos` é a soma do 2º e do 3º maiores picos do espectro; **não** procura múltiplos da frequência fundamental.
- `THD` é uma variante própria (razão entre potência fora do pico e amplitude do pico).
- A `Curtose` é a de Pearson (sinal gaussiano = 3).
- A geometria padrão de `BearingGeometry` (13 esferas de 3,5 mm, diâmetro primitivo de 26 mm) é uma **aproximação** para o
  PRONOSTIA, não conferida em datasheet: use a do seu rolamento. Ela só afeta as 4 features de envelope.
- `SampleEntropy` tem custo O(n²): rápida para janelas de ~2560 pontos, lenta para janelas muito longas.
