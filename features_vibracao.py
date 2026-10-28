"""
features_vibracao.py
====================
Extração de features de um sinal de VIBRAÇÃO (ex.: acelerômetro em um mancal de rolamento) nos domínios do
TEMPO, da FREQUÊNCIA e TEMPO-FREQUÊNCIA. São 75 features por janela de sinal:

    tempo .............. 16   calcular_metricas_tempo(sinal)
    frequência ......... 11   calcular_metricas_frequencia(sinal, fs)
    tempo-frequência ... 48   calcular_metricas_tempo_frequencia(sinal, fs)
                               (STFT: 8 + wavelet DWT db4 em 4 níveis = 5 sub-bandas × 8 = 40)
    todas .............. 75   extrair_todas_features(sinal, fs)
    banco completo ..... 88   extrair_88_features(sinal, fs, geom, rpm)   (+ 13 de features_avancadas.py)

Origem: módulo de features usado no projeto de estimativa de vida útil remanescente (RUL) de rolamentos
(PRONOSTIA / IEEE PHM 2012). O código de cálculo é o MESMO do projeto; aqui foram acrescentados só comentários
e o parâmetro `fs` explícito em extrair_todas_features.

------------------------------------------------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------------------------------------------------
Dependências:  pip install numpy scipy PyWavelets      (veja requirements.txt)

1) Uma janela de sinal (vetor 1D de amplitudes, ex.: aceleração em g) e a frequência de amostragem em Hz:

    import numpy as np
    from features_vibracao import extrair_todas_features

    fs = 25600                                   # Hz (no PRONOSTIA: 25,6 kHz)
    sinal = np.loadtxt("acc_00001.csv", delimiter=",", usecols=4)   # exemplo: canal horizontal do PRONOSTIA
    feats = extrair_todas_features(sinal, fs)    # dict {nome_da_feature: valor}, 75 itens
    print(feats["RMS"], feats["Curtose"], feats["Centroide_Espectral"])

2) Só um domínio:

    from features_vibracao import calcular_metricas_tempo, calcular_metricas_frequencia, \
        calcular_metricas_tempo_frequencia
    tempo = calcular_metricas_tempo(sinal)                  # 16 features
    freq = calcular_metricas_frequencia(sinal, fs)          # 11 features
    tf = calcular_metricas_tempo_frequencia(sinal, fs)      # 48 features

3) Muitas janelas (ex.: um arquivo por medição) -> tabela com uma linha por janela:

    import pandas as pd
    linhas = [extrair_todas_features(x, fs) for x in lista_de_janelas]
    tabela = pd.DataFrame(linhas)                # shape (n_janelas, 75)

Veja também exemplo_uso.py (roda com um sinal sintético, sem precisar de dados).

Observações importantes
- Cada chamada processa UMA janela. As features não dependem de outras janelas (nada de normalização global).
- O sinal deve ser 1D (um canal). Para dois canais (horizontal e vertical), chame uma vez para cada canal.
- As features de frequência usam faixas fixas em Hz (1–5 kHz e corte em 2 kHz): faça sentido com o seu `fs`
  (precisa de fs > 10 kHz para a faixa 1–5 kHz existir inteira).
- Pequenas constantes 1e-12 evitam divisão por zero.
"""

import numpy as np
import pywt
from scipy import signal, stats

FS_PADRAO = 25600   # Hz - frequência de amostragem do PRONOSTIA (use a do seu sensor)


# ============================================================================================================
# 1. DOMÍNIO DO TEMPO (16 features)
# ============================================================================================================
def calcular_metricas_tempo(sinal_):
    """Estatísticas da forma de onda no tempo.

    Entrada: sinal_ (array 1D).  Saída: dict com 16 features:
      Max, Min, Media ............ máximo, mínimo e média do sinal
      Pico ........................ max |x|  (maior amplitude, positiva ou negativa)
      Pico_a_Pico ................. Max - Min
      Desvio_Padrao, Variancia .... dispersão em torno da média
      RMS ......................... sqrt(média(x²)) - energia da vibração (cresce com o dano)
      Potencia .................... média(x²) = RMS²
      Assimetria .................. skewness (simetria da distribuição de amplitudes)
      Curtose ..................... kurtosis de Pearson (= 3 para sinal gaussiano; > 3 indica impactos)
      Coef_Variacao ............... Desvio_Padrao / |Media|  (instável se a média ~ 0)
      Fator_Crista ................ Pico / RMS        (picos isolados em relação à energia)
      Fator_Folga ................. Pico / (média(sqrt|x|))²   ("clearance factor")
      Fator_Forma ................. RMS / média|x|
      Fator_Impulso ............... Pico / média|x|
    """
    m = {}
    m['Max'] = np.max(sinal_)
    m['Min'] = np.min(sinal_)
    m['Media'] = np.mean(sinal_)
    m['Pico'] = np.max(np.abs(sinal_))
    m['Pico_a_Pico'] = m['Max'] - m['Min']
    m['Desvio_Padrao'] = np.std(sinal_)
    m['RMS'] = np.sqrt(np.mean(sinal_ ** 2))
    m['Variancia'] = np.var(sinal_)
    m['Potencia'] = np.mean(sinal_ ** 2)
    m['Assimetria'] = stats.skew(sinal_)
    m['Curtose'] = stats.kurtosis(sinal_, fisher=False)          # fisher=False -> gaussiana = 3
    m['Coef_Variacao'] = m['Desvio_Padrao'] / (abs(m['Media']) + 1e-12)
    m['Fator_Crista'] = m['Pico'] / (m['RMS'] + 1e-12)
    smr = (np.mean(np.sqrt(np.abs(sinal_)))) ** 2                  # "square mean root"
    m['Fator_Folga'] = m['Pico'] / (smr + 1e-12)
    media_abs = np.mean(np.abs(sinal_))
    m['Fator_Forma'] = m['RMS'] / (media_abs + 1e-12)
    m['Fator_Impulso'] = m['Pico'] / (media_abs + 1e-12)
    return m


# ============================================================================================================
# 2. DOMÍNIO DA FREQUÊNCIA (11 features)
# ============================================================================================================
def calcular_metricas_frequencia(sinal_, fs=FS_PADRAO):
    """Features do espectro de potência (FFT de um lado).

    Entrada: sinal_ (array 1D), fs (Hz).  Saída: dict com 11 features. PSD = |FFT|² / N.
      Picos_Espectrais ...... média da PSD nos 3 bins de maior potência
      Bandas_Frequencia ..... soma da PSD entre 1 kHz e 5 kHz (faixa fixa)
      Harmonicos ............ soma da PSD do 2º e do 3º maiores bins
                              (ATENÇÃO: o nome é histórico - NÃO procura múltiplos da frequência fundamental)
      Bandas_Laterais ....... soma da PSD em ±5 bins ao redor do maior pico, sem o próprio pico
      Centroide_Espectral ... frequência média ponderada pela PSD (Hz): "centro de massa" do espectro
      Variancia_Espectral ... dispersão do espectro em torno do centroide (Hz²)
      Assimetria_Espectral .. 3º momento normalizado do espectro
      Curtose_Espectral ..... 4º momento normalizado do espectro
      THD ................... sqrt(PSD total - PSD do maior pico) / |FFT no maior pico|
                              (variante própria de distorção harmônica: mistura potência e amplitude)
      Razoes_Frequencia ..... PSD acima de 2 kHz / PSD abaixo de 2 kHz (corte fixo)
      Cepstro ............... maior |cepstro real| a partir da quefrência de índice 10
                              (detecta periodicidades no espectro, ex.: famílias de impactos)
    """
    m = {}
    N = len(sinal_)
    fft_vals = np.fft.rfft(sinal_)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    psd = np.abs(fft_vals) ** 2 / N

    picos_idx = np.argsort(psd)[-3:]                                  # 3 maiores bins (o último é o maior)
    m['Picos_Espectrais'] = np.mean(psd[picos_idx])
    idx_banda = (freqs >= 1000) & (freqs <= 5000)
    m['Bandas_Frequencia'] = np.sum(psd[idx_banda])
    m['Harmonicos'] = np.sum(psd[picos_idx[:-1]])

    pico_principal = picos_idx[-1]
    idx_laterais = range(max(0, pico_principal - 5), min(len(psd), pico_principal + 6))
    m['Bandas_Laterais'] = np.sum(psd[list(idx_laterais)]) - psd[pico_principal]

    psd_norm = psd / (np.sum(psd) + 1e-12)                            # espectro como distribuição de probabilidade
    centroide = np.sum(freqs * psd_norm)
    m['Centroide_Espectral'] = centroide

    var_esp = np.sum(((freqs - centroide) ** 2) * psd_norm)
    m['Variancia_Espectral'] = var_esp
    std_esp = np.sqrt(var_esp) + 1e-12
    m['Assimetria_Espectral'] = np.sum(((freqs - centroide) ** 3) * psd_norm) / (std_esp ** 3)
    m['Curtose_Espectral'] = np.sum(((freqs - centroide) ** 4) * psd_norm) / (std_esp ** 4)
    m['THD'] = np.sqrt(max(0, np.sum(psd) - psd[pico_principal])) / (np.abs(fft_vals[pico_principal]) + 1e-12)

    b_baixa = np.sum(psd[freqs < 2000]) + 1e-12
    b_alta = np.sum(psd[freqs >= 2000])
    m['Razoes_Frequencia'] = b_alta / b_baixa

    cepstrum = np.fft.irfft(np.log(np.abs(fft_vals) + 1e-12))       # cepstro real
    m['Cepstro'] = np.max(np.abs(cepstrum[10:]))                      # ignora as quefrências mais baixas
    return m


# ============================================================================================================
# 3. DOMÍNIO TEMPO-FREQUÊNCIA (48 features)
# ============================================================================================================
def extrair_stats(vetor, sufixo=""):
    """8 estatísticas de um vetor (usadas na STFT e em cada sub-banda da wavelet).

    Saída: dict com as chaves <estatística>_<sufixo>:
      RMS, Energy (soma de x²), Entropy, Mean, Variance, Std, Skewness, Kurtosis (Pearson, gaussiana = 3).
      Entropy = -Σ p·log2(p) sobre um histograma de 30 classes com density=True
      (as "probabilidades" são densidades do histograma, não frequências relativas: o valor depende da escala).
    """
    counts, _ = np.histogram(vetor, bins=30, density=True)
    counts = counts[counts > 0]
    entropia = -np.sum(counts * np.log2(counts + 1e-12))
    return {
        f'RMS_{sufixo}': np.sqrt(np.mean(vetor ** 2)),
        f'Energy_{sufixo}': np.sum(vetor ** 2),
        f'Entropy_{sufixo}': entropia,
        f'Mean_{sufixo}': np.mean(vetor),
        f'Variance_{sufixo}': np.var(vetor),
        f'Std_{sufixo}': np.std(vetor),
        f'Skewness_{sufixo}': stats.skew(vetor),
        f'Kurtosis_{sufixo}': stats.kurtosis(vetor, fisher=False)
    }


def calcular_metricas_tempo_frequencia(sinal_, fs=FS_PADRAO):
    """Features de tempo-frequência: STFT (8) + wavelet discreta (40) = 48.

    STFT (sufixo "STFT"): espectrograma com janelas de 256 pontos; para cada instante soma-se a magnitude em
      todas as frequências, gerando um vetor ao longo do tempo, e calculam-se as 8 estatísticas desse vetor.
    DWT (sufixos "DWT_L0" ... "DWT_L4"): wavelet Daubechies 4, decomposição em 4 níveis -> 5 sub-bandas:
        DWT_L0 = aproximação A4 (faixa mais baixa)   DWT_L1 = detalhe D4   DWT_L2 = D3
        DWT_L3 = D2                                  DWT_L4 = D1 (faixa mais alta, até fs/2)
      Com fs = 25,6 kHz, D1 cobre ≈ 6,4–12,8 kHz, D2 ≈ 3,2–6,4 kHz, D3 ≈ 1,6–3,2 kHz, D4 ≈ 0,8–1,6 kHz e A4 ≈ 0–0,8 kHz.
      Para cada sub-banda, as 8 estatísticas dos coeficientes.
    """
    mtf = {}
    _, _, Zxx = signal.stft(sinal_, fs=fs, nperseg=256)
    mtf.update(extrair_stats(np.sum(np.abs(Zxx), axis=0), sufixo="STFT"))
    coeffs = pywt.wavedec(sinal_, wavelet='db4', level=4)            # [A4, D4, D3, D2, D1]
    for idx, c in enumerate(coeffs):
        mtf.update(extrair_stats(c, sufixo=f"DWT_L{idx}"))
    return mtf


# ============================================================================================================
# 4. TUDO JUNTO (75 features)
# ============================================================================================================
def extrair_todas_features(sinal_, fs=FS_PADRAO):
    """Junta tempo (16) + frequência (11) + tempo-frequência (48) = 75 features num único dict.

    Entrada: sinal_ (array 1D), fs (Hz).  Saída: dict {nome: valor}.
    """
    m_tempo = calcular_metricas_tempo(sinal_)
    m_freq = calcular_metricas_frequencia(sinal_, fs=fs)
    m_tf = calcular_metricas_tempo_frequencia(sinal_, fs=fs)
    return {**m_tempo, **m_freq, **m_tf}


# ============================================================================================================
# 5. BANCO COMPLETO (88 features) = 75 básicas + 13 avançadas (features_avancadas.py)
# ============================================================================================================
def extrair_88_features(sinal_, fs=FS_PADRAO, geom=None, rpm=None):
    """O banco completo usado no projeto: 75 básicas + entropias, Hjorth, kurtograma e envelope.

    geom: features_avancadas.BearingGeometry do SEU rolamento; rpm: rotação do eixo.
    Com geom e rpm -> 88 features. Sem eles -> 84 (sem as 4 de envelope).

        from features_avancadas import BearingGeometry
        from features_vibracao import extrair_88_features
        feats = extrair_88_features(sinal, 25600, geom=BearingGeometry(), rpm=1800)   # 88 features
    """
    from features_avancadas import extrair_features_avancadas
    feats = extrair_todas_features(sinal_, fs=fs)
    feats.update(extrair_features_avancadas(sinal_, fs, geom, rpm))
    return feats
