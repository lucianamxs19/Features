"""
features_avancadas.py
=====================
13 features "avançadas" de vibração de rolamentos, que completam as 75 de features_vibracao.py e formam o banco
de 88 features usado no projeto de RUL de rolamentos (PRONOSTIA / IEEE PHM 2012):

    entropias ............ 3   PermutationEntropy, SampleEntropy, DispersionEntropy
    Hjorth ............... 3   Hjorth_Activity, Hjorth_Mobility, Hjorth_Complexity
    kurtograma ........... 3   Kurtogram_MaxKurtosis, Kurtogram_BandaOtima_Low, Kurtogram_BandaOtima_High
    envelope ............. 4   Env_BPFO, Env_BPFI, Env_BSF, Env_FTF   (só se a geometria e a rotação forem informadas)

O código de cálculo é o MESMO do projeto (rul_features_extended.py); aqui foram acrescentados só comentários.

------------------------------------------------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------------------------------------------------
    import numpy as np
    from features_avancadas import BearingGeometry, extrair_features_avancadas

    fs = 25600                                   # Hz
    sinal = ...                                  # janela 1D de vibração (ex.: 2560 pontos)

    # sem geometria -> 9 features (entropias, Hjorth, kurtograma)
    av = extrair_features_avancadas(sinal, fs)

    # com geometria do SEU rolamento e a rotação (rpm) -> 13 features (inclui o envelope)
    geom = BearingGeometry(n_balls=13, ball_diameter_mm=3.5, pitch_diameter_mm=26.0, contact_angle_deg=0.0)
    av = extrair_features_avancadas(sinal, fs, geom=geom, rpm=1800)

    # as 88 juntas (75 básicas + 13 avançadas): use extrair_88_features em features_vibracao.py

Observações
- SampleEntropy é O(n²): para 2560 pontos leva alguns décimos de segundo; para janelas grandes fica lenta.
- O kurtograma filtra o sinal em 8 bandas (Butterworth de 4ª ordem); requer fs alto o bastante para a faixa
  500 Hz – 0,45·fs.
- A geometria padrão de BearingGeometry é uma APROXIMAÇÃO (série 6804) e não foi conferida em datasheet: use a do
  rolamento real. Ela só afeta as 4 features de envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq


# ============================================================================================================
# 0. GEOMETRIA DO ROLAMENTO (para as frequências de defeito do envelope)
# ============================================================================================================
@dataclass
class BearingGeometry:
    """Geometria do rolamento -> frequências características de falha.

    n_balls ............ Z, número de elementos rolantes
    ball_diameter_mm ... d, diâmetro da esfera
    pitch_diameter_mm .. D, diâmetro primitivo (circunferência que passa pelo centro das esferas)
    contact_angle_deg .. β, ângulo de contato (0 para rolamento radial de esferas)

    Os valores padrão são uma aproximação típica da série 6804 (usada como referência para o PRONOSTIA) e NÃO
    foram confirmados no datasheet. Para outro rolamento, informe os valores reais.
    """
    n_balls: int = 13
    ball_diameter_mm: float = 3.5
    pitch_diameter_mm: float = 26.0
    contact_angle_deg: float = 0.0

    def characteristic_frequencies(self, fr_hz: float) -> dict:
        """fr_hz = rotação do eixo em Hz (rpm / 60). Devolve BPFO, BPFI, BSF e FTF em Hz:
            BPFO = (Z/2)·fr·(1 − (d/D)cos β)       defeito na pista EXTERNA
            BPFI = (Z/2)·fr·(1 + (d/D)cos β)       defeito na pista INTERNA
            BSF  = (D/2d)·fr·(1 − ((d/D)cos β)²)   defeito na ESFERA (frequência de giro da esfera)
            FTF  = (fr/2)·(1 − (d/D)cos β)         defeito na GAIOLA
        """
        Z, Db, Dp = self.n_balls, self.ball_diameter_mm, self.pitch_diameter_mm
        beta = np.radians(self.contact_angle_deg)
        ratio = (Db / Dp) * np.cos(beta)

        bpfo = (Z / 2) * fr_hz * (1 - ratio)
        bpfi = (Z / 2) * fr_hz * (1 + ratio)
        bsf = (Dp / (2 * Db)) * fr_hz * (1 - ratio ** 2)
        ftf = (fr_hz / 2) * (1 - ratio)
        return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


# ============================================================================================================
# 1. ENVELOPE (HILBERT) NAS FREQUÊNCIAS DE FALHA - 4 features
# ============================================================================================================
def envelope_spectrum(sinal, fs):
    """Espectro do envelope: |sinal analítico (Hilbert)| sem a média -> FFT. Os impactos de um defeito
    modulam a vibração; a demodulação revela o RITMO desses impactos (as frequências de defeito)."""
    analytic = signal.hilbert(sinal)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)
    N = len(envelope)
    spec = np.abs(rfft(envelope)) / N
    freqs = rfftfreq(N, d=1 / fs)
    return freqs, spec


def amplitude_em_frequencia(freqs, spec, f_alvo, banda_hz=5.0, n_harmonicos=3):
    """Soma, para as 3 primeiras harmônicas de f_alvo (f, 2f, 3f), o MAIOR valor do espectro de envelope numa
    faixa de ±5 Hz em torno de cada uma."""
    total = 0.0
    for h in range(1, n_harmonicos + 1):
        f_h = f_alvo * h
        idx = (freqs >= f_h - banda_hz) & (freqs <= f_h + banda_hz)
        if np.any(idx):
            total += np.max(spec[idx])
    return total


def features_envelope_rolamento(sinal, fs, geom: BearingGeometry, rpm: float):
    """Env_BPFO, Env_BPFI, Env_BSF, Env_FTF: amplitude do espectro de envelope em cada frequência de defeito.
    Cresce quando o tipo de defeito correspondente aparece. Precisa da geometria e da rotação (rpm)."""
    fr_hz = rpm / 60.0
    freqs_carac = geom.characteristic_frequencies(fr_hz)
    freqs, spec = envelope_spectrum(sinal, fs)

    feats = {}
    for nome, f_alvo in freqs_carac.items():
        feats[f"Env_{nome}"] = amplitude_em_frequencia(freqs, spec, f_alvo)
    return feats


# ============================================================================================================
# 2. KURTOGRAMA SIMPLIFICADO - 3 features
# ============================================================================================================
def kurtogram_simplificado(sinal, fs, n_bandas=8, banda_min=500, banda_max=None):
    """Versão simplificada do kurtograma de Antoni: divide 500 Hz – 0,45·fs em 8 bandas iguais, filtra cada uma
    (Butterworth passa-banda de 4ª ordem, sem atraso), calcula a curtose do envelope e fica com a maior.

    Kurtogram_MaxKurtosis ....... curtose do envelope na melhor banda (impactos mais visíveis -> valor maior)
    Kurtogram_BandaOtima_Low/High limites (Hz) dessa banda (indicam ONDE está o impacto, não o quanto)
    """
    if banda_max is None:
        banda_max = fs / 2 * 0.9

    limites = np.linspace(banda_min, banda_max, n_bandas + 1)
    melhor_curtose, melhor_banda = -np.inf, (banda_min, banda_max)

    for i in range(n_bandas):
        f_low, f_high = limites[i], limites[i + 1]
        sos = signal.butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        filtrado = signal.sosfiltfilt(sos, sinal)
        env = np.abs(signal.hilbert(filtrado))
        kurt = stats.kurtosis(env, fisher=False)
        if kurt > melhor_curtose:
            melhor_curtose, melhor_banda = kurt, (f_low, f_high)

    return {"Kurtogram_MaxKurtosis": melhor_curtose,
            "Kurtogram_BandaOtima_Low": melhor_banda[0],
            "Kurtogram_BandaOtima_High": melhor_banda[1]}


# ============================================================================================================
# 3. ENTROPIAS - 3 features (medem a "complexidade"/regularidade do sinal)
# ============================================================================================================
def permutation_entropy(sinal, m=5, tau=1):
    """Entropia de permutação (Bandt & Pompe), normalizada entre 0 e 1: conta os padrões de ORDEM de m pontos
    consecutivos (ex.: sobe-sobe-desce). Perto de 1 = sinal irregular; menor = sinal mais regular/estruturado."""
    n = len(sinal)
    permutations = {}
    for i in range(n - (m - 1) * tau):
        janela = sinal[i: i + m * tau: tau]
        ordem = tuple(np.argsort(janela))
        permutations[ordem] = permutations.get(ordem, 0) + 1

    total = sum(permutations.values())
    probs = np.array([v / total for v in permutations.values()])
    pe = -np.sum(probs * np.log2(probs))
    pe_norm = pe / np.log2(math.factorial(m))
    return pe_norm


def sample_entropy(sinal, m=2, r=None):
    """Sample entropy (SampEn): −ln(A/B), em que B e A contam pares de trechos de m e m+1 pontos parecidos
    (distância máxima ≤ r = 0,2·desvio-padrão). Valor maior = sinal menos previsível. Custo O(n²)."""
    sinal = np.asarray(sinal, dtype=float)
    n = len(sinal)
    if r is None:
        r = 0.2 * np.std(sinal)

    def _count_matches(m_):
        templates = np.array([sinal[i: i + m_] for i in range(n - m_)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1) if i < len(templates) - 1 else np.array([])
            count += np.sum(dist <= r)
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def dispersion_entropy(sinal, m=3, c=6, tau=1):
    """Entropia de dispersão (Rostaghi & Azami), normalizada entre 0 e 1: mapeia cada ponto em c = 6 classes
    (pela CDF normal) e mede a diversidade dos padrões de m = 3 classes consecutivas."""
    n = len(sinal)
    mu, sigma = np.mean(sinal), np.std(sinal) + 1e-12
    z = stats.norm.cdf((sinal - mu) / sigma)
    y = np.round(c * z + 0.5).clip(1, c).astype(int)

    padroes = {}
    for i in range(n - (m - 1) * tau):
        padrao = tuple(y[i: i + m * tau: tau])
        padroes[padrao] = padroes.get(padrao, 0) + 1

    total = sum(padroes.values())
    probs = np.array([v / total for v in padroes.values()])
    de = -np.sum(probs * np.log2(probs))
    de_norm = de / np.log2(c ** m)
    return de_norm


def features_entropia(sinal, m=3, tau=1):
    """As 3 entropias com os parâmetros usados no projeto (permutação m = 5; amostra m = 2; dispersão m = 3, c = 6)."""
    return {
        "PermutationEntropy": permutation_entropy(sinal, m=5, tau=tau),
        "SampleEntropy": sample_entropy(sinal, m=2),
        "DispersionEntropy": dispersion_entropy(sinal, m=m, c=6, tau=tau),
    }


# ============================================================================================================
# 4. PARÂMETROS DE HJORTH - 3 features
# ============================================================================================================
def hjorth_parameters(sinal):
    """Hjorth_Activity = variância do sinal (potência)
    Hjorth_Mobility   = sqrt(var(x') / var(x))       ~ frequência média
    Hjorth_Complexity = Mobility(x') / Mobility(x)    ~ quanto o sinal se afasta de uma senoide pura (= 1)
    (x' e x'' são as diferenças de 1ª e 2ª ordem do sinal)."""
    d1 = np.diff(sinal)
    d2 = np.diff(d1)
    var0, var1, var2 = np.var(sinal), np.var(d1), np.var(d2)

    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)
    return {"Hjorth_Activity": activity, "Hjorth_Mobility": mobility,
            "Hjorth_Complexity": complexity}


# ============================================================================================================
# 5. TUDO JUNTO
# ============================================================================================================
def extrair_features_avancadas(sinal, fs, geom: BearingGeometry = None, rpm: float = None):
    """9 features (entropias, Hjorth, kurtograma) ou 13 (+ envelope) se `geom` e `rpm` forem informados."""
    feats = {}
    feats.update(features_entropia(sinal))
    feats.update(hjorth_parameters(sinal))
    feats.update(kurtogram_simplificado(sinal, fs))
    if geom is not None and rpm is not None:
        feats.update(features_envelope_rolamento(sinal, fs, geom, rpm))
    return feats
