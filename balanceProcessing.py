import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import interpolate, signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch, detrend, savgol_filter
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.stats import linregress
import streamlit as st

# ====================== Configuração da página ======================
st.set_page_config(page_title="Análise Postural", layout="wide", page_icon="📊")
st.title("📊 Análise de Equilíbrio Postural")
st.markdown(
    "Aplicativo para análise de dados de **acelerômetro** para avaliação do equilíbrio postural.")

# ====================== Sidebar - Controles principais ======================
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader(
        "Carregar arquivo de dados (.csv ou .txt)", type=['txt', 'csv'])

    st.subheader("Parâmetros de Filtro")
    cutoff_freq = st.slider("Frequência de corte (Hz)", 1.0, 20.0, 6.0)
    filter_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1)

    st.subheader("Seleção de Intervalo")
    clip_signal = st.checkbox("Clipar sinal para análise", False)
    start_time = st.number_input("Tempo inicial (s)", 0.0, 1e6, 0.0)
    end_time = st.number_input("Tempo final (s)", 0.0, 1e6, 30.0)

    st.subheader("Visualizações")
    show_raw = st.checkbox("Mostrar dados brutos", True)
    show_filtered = st.checkbox("Mostrar dados filtrados", True)
    show_velocity = st.checkbox("Mostrar velocidade", False)
    show_displacement = st.checkbox("Mostrar deslocamento", False)

# ====================== Utilidades de I/O ======================


def infer_sep_and_read(file) -> Tuple[pd.DataFrame, str]:
    """Tenta ler CSV inferindo separador: ',', ';', '\\t' (ou auto)."""
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(file, sep=sep)
            if df.shape[1] >= 2:
                return df, sep
        except Exception:
            pass
    df = pd.read_csv(file, sep=None, engine="python")
    return df, "infer"


def map_acc_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Encontra colunas (tempo, accAP, accML, accZ) com alternativas comuns."""
    # tempo
    time_col = None
    for c in ["Tempo", "Time (s)", "time", "Time", "timestamp", "t"]:
        if c in df.columns:
            time_col = c
            break

    # acelerômetro (nomes do seu arquivo e alternativas)
    ap_col = None
    ml_col = None
    z_col = None

    if "accAP" in df.columns:
        ap_col = "accAP"
    if "accML" in df.columns:
        ml_col = "accML"
    if "accZ" in df.columns:
        z_col = "accZ"

    cand_ap = ["accAP", "Ax", "ax", "AccX", "accX", "X", "x"]
    cand_ml = ["accML", "Ay", "ay", "AccY", "accY", "Y", "y"]
    cand_z = ["accZ", "Az", "az", "AccZ", "acc_z", "Z", "z"]

    if ap_col is None:
        for c in cand_ap:
            if c in df.columns:
                ap_col = c
                break
    if ml_col is None:
        for c in cand_ml:
            if c in df.columns:
                ml_col = c
                break
    if z_col is None:
        for c in cand_z:
            if c in df.columns:
                z_col = c
                break

    return time_col, ap_col, ml_col, z_col

# ====================== Funções do seu código original (mantidas/ajustadas) ======================


def to_numpy(column):
    return column.to_numpy()


def tirar_gravidade(dado1: np.ndarray, dado2: np.ndarray):
    # Heurística: se um eixo tem ~9.8 m/s², usamos o outro como ML
    return dado2 if np.nanmax(np.abs(dado1)) > 9 else dado1


def calculate_ellipse(ml_acc: np.ndarray, ap_acc: np.ndarray, confidence: float = 0.95) -> Tuple:
    cov = np.cov(ml_acc, ap_acc)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(vals * 5.991)  # CHI2 95%
    principal_dir = np.cos(
        abs(vecs[1, 0]) / np.linalg.norm(vecs[:, 0])) * 180 / np.pi
    area = np.pi * scale[0] * scale[1]
    return np.mean(ml_acc), np.mean(ap_acc), scale[0]*2, scale[1]*2, angle, principal_dir, area


def calculate_metrics(ml_acc: np.ndarray, ap_acc: np.ndarray) -> Tuple:
    rms_ml = np.sqrt(np.mean(ml_acc**2))
    rms_ap = np.sqrt(np.mean(ap_acc**2))
    total_deviation = np.sum(np.sqrt(ml_acc**2 + ap_acc**2))
    _, _, width, height, _, _, _ = calculate_ellipse(ml_acc, ap_acc)
    ellipse_area = np.pi * width * height / 4
    return rms_ml, rms_ap, total_deviation, ellipse_area


def spectrum_plot(ml, ap, fs):
    n = len(ml)
    fft_ml = np.fft.fft(ml)
    fft_ap = np.fft.fft(ap)
    freqs = np.fft.fftfreq(n, d=1/fs)
    positive_freqs = freqs[:n//2]
    psd_ml = np.abs(fft_ml[:n//2])**2 / (n*fs)
    psd_ap = np.abs(fft_ap[:n//2])**2 / (n*fs)
    return positive_freqs, psd_ml, psd_ap


def analyze_balance(ml, ap, sampling_rate=100):
    n = len(ml)
    R = np.sqrt(ml**2 + ap**2)
    cov = np.mean(ml * ap)  # não usado adiante, mantido por compatibilidade

    rms_ml = np.sqrt(np.mean(ml**2))
    rms_ap = np.sqrt(np.mean(ap**2))
    planar_dev = np.sqrt(np.std(ml)**2 + np.std(ap)**2)
    sway_dir = np.arctan2(np.std(ap), np.std(ml))

    _, _, _, _, _, _, area = calculate_ellipse(ml, ap)
    freqs, psd_ml, psd_ap = spectrum_plot(ml, ap, sampling_rate)

    total_power_ml = trapezoid(psd_ml, freqs)
    total_power_ap = trapezoid(psd_ap, freqs)

    bands = [(0, 0.5), (0.5, 2), (2, np.inf)]
    energy_ml = [trapezoid(psd_ml[(freqs >= b[0]) & (freqs < b[1])], freqs[(
        freqs >= b[0]) & (freqs < b[1])]) for b in bands]
    energy_ap = [trapezoid(psd_ap[(freqs >= b[0]) & (freqs < b[1])], freqs[(
        freqs >= b[0]) & (freqs < b[1])]) for b in bands]

    centroid_ml = trapezoid(freqs * psd_ml, freqs) / \
        (total_power_ml if total_power_ml > 0 else 1)
    centroid_ap = trapezoid(freqs * psd_ap, freqs) / \
        (total_power_ap if total_power_ap > 0 else 1)

    mode_ml = freqs[np.argmax(psd_ml)]
    mode_ap = freqs[np.argmax(psd_ap)]

    disp_ml = np.sqrt(trapezoid(((freqs - centroid_ml) ** 2) * psd_ml,
                      freqs) / (total_power_ml if total_power_ml > 0 else 1))
    disp_ap = np.sqrt(trapezoid(((freqs - centroid_ap) ** 2) * psd_ap,
                      freqs) / (total_power_ap if total_power_ap > 0 else 1))

    _, _, total_dev, ellipse_area = calculate_metrics(ml, ap)

    return [
        rms_ap, rms_ml, area, planar_dev, sway_dir, total_power_ml, total_power_ap,
        *energy_ml, *energy_ap, centroid_ml, centroid_ap,
        mode_ml, mode_ap, disp_ml, disp_ap, round(total_dev, 4)
    ]


def calcular_velocidade(acc, fs, metodo='savitzky', cutoff=5, ordem=3, janela=21):
    dt = 1 / fs
    tempo = np.arange(len(acc)) * dt

    # garantir janela ímpar e <= N
    N = len(acc)
    if janela >= N:
        janela = max(3, (N//2)*2 - 1)  # maior ímpar < N
    if janela % 2 == 0:
        janela += 1
    ordem = min(ordem, max(1, janela - 2))

    if metodo == 'savitzky':
        acc_filtrado = savgol_filter(acc, janela, ordem)
    elif metodo == 'butterworth':
        b, a = butter(ordem, cutoff / (fs / 2), btype='low')
        acc_filtrado = filtfilt(b, a, acc)
    elif metodo == 'spline':
        spl = interpolate.UnivariateSpline(tempo, acc, k=min(5, ordem))
        acc_filtrado = spl(tempo)
    else:
        raise ValueError("Método desconhecido")

    velocidade = cumulative_trapezoid(acc_filtrado, dx=dt, initial=0)
    velocidade = detrend(velocidade, type='linear')
    return tempo, velocidade


def highpass_filter(sig, fc, fs, order=4):
    w = fc / (fs / 2)
    b, a = butter(order, w, btype='high')
    return filtfilt(b, a, sig)


def zero_crossings(sig):
    sig = np.asarray(sig)
    return len(np.where(np.diff(np.sign(sig)))[0])


def analise_difusao(cop_x, cop_y, fs, max_lag_seg=5):
    max_lag = int(max_lag_seg * fs)
    lags = np.arange(1, max_lag)
    msd_x = np.array([np.mean((cop_x[lag:] - cop_x[:-lag]) ** 2)
                     for lag in lags])
    msd_y = np.array([np.mean((cop_y[lag:] - cop_y[:-lag]) ** 2)
                     for lag in lags])
    return lags / fs, msd_x + msd_y, msd_x, msd_y


def extrair_parametros_difusao(time_lags, msd_total):
    curto = time_lags <= 0.5
    longo = time_lags >= 1.5
    slope1, intercept1, *_ = linregress(time_lags[curto], msd_total[curto])
    slope2, intercept2, *_ = linregress(time_lags[longo], msd_total[longo])
    cp = (intercept2 - intercept1) / (slope1 - slope2)
    sway_cp = slope1 * cp + intercept1
    values = [slope1, slope2, intercept1, intercept2, cp, sway_cp]
    return [float(v) for v in values]


# ====================== Leitura do arquivo (uploader OU anexo) ======================
df_in = None
src_info = ""

if uploaded_file is not None:
    df_in, sep = infer_sep_and_read(uploaded_file)
    src_info = f"Arquivo carregado via uploader (sep='{sep}')"
else:
    default_path = "/mnt/data/ACC_AP_ML_Z.csv"
    if os.path.exists(default_path):
        df_in, sep = infer_sep_and_read(default_path)
        src_info = f"Abrido do anexo: {default_path} (sep='{sep}')"
    else:
        st.info(
            "Envie um arquivo (.csv ou .txt) ou disponibilize o anexo em /mnt/data/ACC_AP_ML_Z.csv.")
        st.stop()

st.caption(src_info)
st.dataframe(df_in.head(10), use_container_width=True)

# ====================== Mapeamento/normalização de colunas ======================
time_col, ap_col, ml_col, z_col = map_acc_columns(df_in)
missing = []
if time_col is None:
    missing.append("Tempo / Time (s)")
if ap_col is None:
    missing.append("accAP / Ax")
if ml_col is None:
    missing.append("accML / Ay")
if z_col is None:
    missing.append("accZ / Az")
if missing:
    st.error("Não encontrei as colunas: " + ", ".join(missing))
    st.stop()

# força numérico e remove NaNs
for c in [time_col, ap_col, ml_col, z_col]:
    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
df_in = df_in.dropna(subset=[time_col, ap_col, ml_col, z_col]).copy()

# ====================== Tempo → segundos (auto) ======================
tempo_raw = df_in[time_col].to_numpy()
# Heurística: se valores são grandes (ex. > 1e4) são ms → converter
t_is_ms = np.nanmax(tempo_raw) > 1e4
t_original = tempo_raw / 1000.0 if t_is_ms else tempo_raw

# ====================== Sinais originais ======================
a_ap = df_in[ap_col].to_numpy()
a_ml = df_in[ml_col].to_numpy()
a_z = df_in[z_col].to_numpy()

# ML seguindo sua heurística de gravidade e AP como canal AP
ap = a_ap
ml = tirar_gravidade(a_z, a_ml)

# ====================== Estima fs e reamostra para 100Hz ======================
dts = np.diff(t_original)
dts = dts[dts > 0]
if dts.size == 0:
    st.error("Não foi possível estimar a taxa de amostragem pelo vetor de tempo.")
    st.stop()
fs_est = 1.0 / np.median(dts)
st.success(
    f"Taxa de amostragem estimada: **{fs_est:.3f} Hz** (tempo {'ms' if t_is_ms else 's'})")

fs_novo = 100.0
dt_novo = 1.0 / fs_novo
t_novo = np.arange(t_original[0], t_original[-1] + 1e-9, dt_novo)

interp_ap = interp1d(t_original, ap, kind='linear',
                     fill_value="extrapolate", assume_sorted=True)
interp_ml = interp1d(t_original, ml, kind='linear',
                     fill_value="extrapolate", assume_sorted=True)
ap_interp_100Hz = interp_ap(t_novo)
ml_interp_100Hz = interp_ml(t_novo)

# ====================== Detrend + Filtro passa-baixa ======================
ap_detrended = detrend(ap_interp_100Hz)
ml_detrended = detrend(ml_interp_100Hz)

fs = fs_novo
nyquist = fs / 2.0
normal_cutoff = min(0.99, float(cutoff_freq) / nyquist)  # sanidade

b, a = butter(int(filter_order), normal_cutoff, btype='low', analog=False)
ap_filtrado = filtfilt(b, a, ap_detrended)
ml_filtrado = filtfilt(b, a, ml_detrended)

# ====================== Clip por tempo (opcional) ======================
if clip_signal:
    t0 = max(t_novo[0], start_time)
    t1 = min(t_novo[-1], end_time if end_time > start_time else t_novo[-1])
else:
    t0, t1 = t_novo[0], t_novo[-1]

mask = (t_novo >= t0) & (t_novo <= t1)
t_novo_clipped = t_novo[mask]
ap_filtrado = ap_filtrado[mask]
ml_filtrado = ml_filtrado[mask]

# ====================== Velocidade e deslocamento ======================
time_vec_vel_ap, vel_ap = calcular_velocidade(ap_filtrado, fs)
time_vec_vel_ml, vel_ml = calcular_velocidade(ml_filtrado, fs)

deslocamento_bruto_ap = cumulative_trapezoid(vel_ap, dx=1/fs, initial=0)
deslocamento_bruto_ml = cumulative_trapezoid(vel_ml, dx=1/fs, initial=0)

fc_hp = 0.1  # Hz
deslocamento_corrigido_ap = highpass_filter(
    deslocamento_bruto_ap, fc=fc_hp, fs=fs)
deslocamento_corrigido_ml = highpass_filter(
    deslocamento_bruto_ml, fc=fc_hp, fs=fs)

# ====================== Features ======================
features_aceleracao = analyze_balance(ml_filtrado, ap_filtrado, int(fs))

lags_disp, msd_total_disp, msd_ap_disp, msd_ml_disp = analise_difusao(
    deslocamento_corrigido_ap, deslocamento_corrigido_ml, int(fs))
lags_accel, msd_total_accel, msd_ap_accel, msd_ml_accel = analise_difusao(
    ap_filtrado, ml_filtrado, int(fs))

difusao_central_disp = extrair_parametros_difusao(lags_disp, msd_total_disp)
difusao_ap_disp = extrair_parametros_difusao(lags_disp, msd_ap_disp)
difusao_ml_disp = extrair_parametros_difusao(lags_disp, msd_ml_disp)

difusao_central_accel = extrair_parametros_difusao(lags_accel, msd_total_accel)
difusao_ap_accel = extrair_parametros_difusao(lags_accel, msd_ap_accel)
difusao_ml_accel = extrair_parametros_difusao(lags_accel, msd_ml_accel)

zero_ap = zero_crossings(vel_ap)
zero_ml = zero_crossings(vel_ml)

# ====================== Visualizações ======================
tab1, tab2, tab3 = st.tabs(["📈 Gráficos", "📊 Métricas", "💾 Exportar"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        if show_raw:
            st.subheader("Dados Brutos (reamostrados 100 Hz)")
            fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
            ax_raw.plot(t_novo_clipped,
                        ap_interp_100Hz[mask], label='AP bruto', alpha=0.5)
            ax_raw.plot(t_novo_clipped,
                        ml_interp_100Hz[mask], label='ML bruto', alpha=0.5)
            ax_raw.set_xlabel("Tempo (s)")
            ax_raw.set_ylabel("Aceleração (m/s²)")
            ax_raw.legend()
            ax_raw.grid(True, alpha=0.3)
            st.pyplot(fig_raw)

        if show_filtered:
            st.subheader("Aceleração Filtrada")
            fig_ap, ax_ap = plt.subplots(figsize=(10, 4))
            ax_ap.plot(t_novo_clipped, ap_filtrado,
                       label='AP Filtrado', color='blue')
            ax_ap.set_xlabel("Tempo (s)")
            ax_ap.set_ylabel("Aceleração AP (m/s²)")
            ax_ap.legend()
            ax_ap.grid(True, alpha=0.3)
            st.pyplot(fig_ap)

            fig_ml, ax_ml = plt.subplots(figsize=(10, 4))
            ax_ml.plot(t_novo_clipped, ml_filtrado,
                       label='ML Filtrado', color='orange')
            ax_ml.set_xlabel("Tempo (s)")
            ax_ml.set_ylabel("Aceleração ML (m/s²)")
            ax_ml.legend()
            ax_ml.grid(True, alpha=0.3)
            st.pyplot(fig_ml)

        st.subheader("Elipse de Confiança (95%)")
        fig_ellipse, ax_ellipse = plt.subplots(figsize=(6, 6))
        mean_ml, mean_ap, width, height, angle, _, _ = calculate_ellipse(
            ml_filtrado, ap_filtrado)
        ellipse = Ellipse(xy=(mean_ml, mean_ap), width=width,
                          height=height, angle=angle, alpha=0.3, color='blue')
        ax_ellipse.add_patch(ellipse)
        ax_ellipse.scatter(ml_filtrado, ap_filtrado, s=1, alpha=0.5)
        # limites automáticos com margem
        pad = 0.1 * max(np.std(ml_filtrado), np.std(ap_filtrado), 1e-6)
        ax_ellipse.set_xlim(np.min(ml_filtrado)-pad, np.max(ml_filtrado)+pad)
        ax_ellipse.set_ylim(np.min(ap_filtrado)-pad, np.max(ap_filtrado)+pad)
        ax_ellipse.axhline(0, color='black', linestyle='--',
                           linewidth=0.5, alpha=0.5)
        ax_ellipse.axvline(0, color='black', linestyle='--',
                           linewidth=0.5, alpha=0.5)
        ax_ellipse.set_xlabel("ML (m/s²)")
        ax_ellipse.set_ylabel("AP (m/s²)")
        ax_ellipse.set_title("Elipse de Confiança 95%")
        st.pyplot(fig_ellipse)

    with col2:
        if show_velocity:
            st.subheader("Velocidade")
            fig_vel, ax_vel = plt.subplots(figsize=(10, 4))
            ax_vel.plot(time_vec_vel_ap, vel_ap, label='AP')
            ax_vel.plot(time_vec_vel_ml, vel_ml, label='ML')
            ax_vel.set_xlabel("Tempo (s)")
            ax_vel.set_ylabel("Velocidade (m/s)")
            ax_vel.legend()
            ax_vel.grid(True, alpha=0.3)
            st.pyplot(fig_vel)

        if show_displacement:
            st.subheader("Deslocamento")
            fig_disp, ax_disp = plt.subplots(figsize=(10, 4))
            ax_disp.plot(time_vec_vel_ap,
                         deslocamento_corrigido_ap, label='AP')
            ax_disp.plot(time_vec_vel_ml,
                         deslocamento_corrigido_ml, label='ML')
            ax_disp.set_xlabel("Tempo (s)")
            ax_disp.set_ylabel("Deslocamento (m)")
            ax_disp.legend()
            ax_disp.grid(True, alpha=0.3)
            st.pyplot(fig_disp)

        freqs, psd_ml, psd_ap = spectrum_plot(
            ml_filtrado, ap_filtrado, int(fs))
        fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
        ax_psd.plot(freqs, psd_ml, label='ML')
        ax_psd.set_xlim(0, 8)
        ax_psd.set_xlabel("Frequência (Hz)")
        ax_psd.set_ylabel("Densidade Espectral de Potência ML")
        ax_psd.legend()
        ax_psd.grid(True, alpha=0.3)
        st.pyplot(fig_psd)

        fig_psd_ap, ax_psd_ap = plt.subplots(figsize=(10, 4))
        ax_psd_ap.plot(freqs, psd_ap, label='AP')
        ax_psd_ap.set_xlim(0, 8)
        ax_psd_ap.set_xlabel("Frequência (Hz)")
        ax_psd_ap.set_ylabel("Densidade Espectral de Potência AP")
        ax_psd_ap.legend()
        ax_psd_ap.grid(True, alpha=0.3)
        st.pyplot(fig_psd_ap)

        st.subheader("Análise de Difusão — MSD")
        fig_msd_x, ax_msd_x = plt.subplots(figsize=(8, 4))
        ax_msd_x.plot(lags_disp, msd_ap_disp, color='blue')
        ax_msd_x.set_xlabel("Lag (s)")
        ax_msd_x.set_ylabel("MSD - AP")
        ax_msd_x.set_title("MSD — Direção AP (deslocamento)")
        ax_msd_x.grid(True)
        st.pyplot(fig_msd_x)

        fig_msd_y, ax_msd_y = plt.subplots(figsize=(8, 4))
        ax_msd_y.plot(lags_disp, msd_ml_disp, color='green')
        ax_msd_y.set_xlabel("Lag (s)")
        ax_msd_y.set_ylabel("MSD - ML")
        ax_msd_y.set_title("MSD — Direção ML (deslocamento)")
        ax_msd_y.grid(True)
        st.pyplot(fig_msd_y)

with tab2:
    st.subheader("Métricas de Aceleração")
    feature_names = [
        "RMS AP", "RMS ML", "Área da Elipse", "Desvio Planar",
        "Direção da Oscilação", "Potência Total ML", "Potência Total AP",
        "Energia ML 0–0.5 Hz", "Energia ML 0.5–2 Hz", "Energia ML >2 Hz",
        "Energia AP 0–0.5 Hz", "Energia AP 0.5–2 Hz", "Energia AP >2 Hz",
        "Centroide ML", "Centroide AP", "Moda ML", "Moda AP",
        "Dispersão ML", "Dispersão AP", "Desvio Total"
    ]
    cols = st.columns(3)
    for i, (name, value) in enumerate(zip(feature_names, features_aceleracao)):
        with cols[i % 3]:
            st.metric(label=name, value=f"{value:.4f}")

    st.subheader("Análise de Difusão — Deslocamento")
    diff_names = ["H1", "H2", "Intercept1",
                  "Intercept2", "Ponto Crítico", "Sway no CP"]

    st.write("**Central (AP+ML):**")
    cols = st.columns(6)
    for i, (name, value) in enumerate(zip(diff_names, difusao_central_disp)):
        with cols[i]:
            st.metric(label=name, value=f"{value:.6f}")

    st.write("**AP:**")
    cols = st.columns(6)
    for i, (name, value) in enumerate(zip(diff_names, difusao_ap_disp)):
        with cols[i]:
            st.metric(label=name, value=f"{value:.6f}")

    st.write("**ML:**")
    cols = st.columns(6)
    for i, (name, value) in enumerate(zip(diff_names, difusao_ml_disp)):
        with cols[i]:
            st.metric(label=name, value=f"{value:.6f}")

    st.subheader("Cruzamentos por Zero")
    cols = st.columns(2)
    with cols[0]:
        st.metric(label="Cruzamentos AP", value=zero_ap)
    with cols[1]:
        st.metric(label="Cruzamentos ML", value=zero_ml)

with tab3:
    st.subheader("Exportar Resultados")
    all_results = {
        "Métrica": feature_names
        + [f"Difusão Central {n}" for n in diff_names]
        + [f"Difusão AP {n}" for n in diff_names]
        + [f"Difusão ML {n}" for n in diff_names]
        + ["Cruzamentos AP", "Cruzamentos ML"],
        "Valor": features_aceleracao
        + difusao_central_disp
        + difusao_ap_disp
        + difusao_ml_disp
        + [zero_ap, zero_ml]
    }
    results_df = pd.DataFrame(all_results)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar resultados como CSV",
        data=csv,
        file_name="resultados_equilibrio.csv",
        mime="text/csv"
    )
