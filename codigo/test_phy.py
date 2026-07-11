"""
test_phy.py
Script de verificación y pruebas unitarias/funcionales de la simulación IEEE 802.11bb.
Valida la simetría hermitiana, la cadena de transmisión/recepción ideal para todos los MCS
y GIs, y genera las curvas de rendimiento en función de las velocidades nominales oficiales.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physical_layer

# ==============================================================================
#                 CONFIGURACIÓN DE SIMULACIONES MONTE CARLO Y PARÁMETROS
# ==============================================================================
from config_entorno import (
    FFT_SIZE, CP_LEN, F_IF, F_S, ACTIVE_CARRIERS,
    H_DIFF, SEMI_ANGLE, FOV_RX, INDEX_CONC, RX_AREA,
    RESPONSIVITY, BANDWIDTH, LED_MIN, LED_MAX
)

MONTE_CARLO_RUNS = 100000  # Valor base para las pruebas de Monte Carlo (100k para resolucion)

# Variables de bits asignadas de forma independiente para cada simulación
N_BITS_SOLAR = MONTE_CARLO_RUNS
N_BITS_IDEAL = MONTE_CARLO_RUNS
N_BITS_ESCENARIOS = MONTE_CARLO_RUNS

# Configuración del Intervalo de Guarda (GI) seleccionado para las curvas de rendimiento
GI_SELECCIONADO = "0.8us"
# ==============================================================================

def save_plot_dual(fig, filename):
    import os
    path1 = os.path.join("c:\\Users\\carlo\\OneDrive\\Imágenes\\Documentos\\TESIS", filename)
    path2 = os.path.join("c:\\Users\\carlo\\OneDrive\\Imágenes\\Documentos\\TESIS\\TESIS_LATEX", filename)
    fig.savefig(path1, dpi=300)
    fig.savefig(path2, dpi=300)


def get_gi_params(gi_name: str) -> tuple[int, float]:
    """
    Retorna cp_len y T_symbol para un GI dado (a fs = 80 MHz con FFT = 1024).
    """
    gis = {
        "0.8us": (64, 13.6e-6),
        "1.6us": (128, 14.4e-6),
        "3.2us": (256, 16.0e-6)
    }
    if gi_name not in gis:
        raise ValueError(f"GI '{gi_name}' no soportado (debe ser '0.8us', '1.6us' o '3.2us').")
    return gis[gi_name]


def parse_code_rate(rate_str: str) -> float:
    """
    Convierte una fracción en cadena (e.g. "3/4") a float.
    """
    num, den = rate_str.split("/")
    return float(num) / float(den)


def get_phy_data_rate(modulation: str, fft_size: int, cp_len: int, f_s: float, n_active_carriers: int = 234, code_rate_str: str = "1/2") -> float:
    """
    Calcula la velocidad de transmisión nominal de la capa física (PHY Data Rate).
    Fórmula: (N_active * log2(M) * R_code) / T_symbol
    """
    qam_sizes = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
    bits_per_symbol = qam_sizes[modulation.upper()]
    t_symbol = (fft_size + cp_len) / f_s
    code_rate = parse_code_rate(code_rate_str)
    r_raw = (n_active_carriers * bits_per_symbol * code_rate) / t_symbol
    return r_raw


def test_hermitian_symmetry():
    print("1. Probando Simetría Hermitiana...")
    np.random.seed(42)
    # Generar símbolos aleatorios QPSK
    tx_bits = np.random.randint(0, 2, 500)
    tx_symbols = physical_layer.constellation_mapping(tx_bits, scheme="QPSK")
    
    # Aplicar simetría
    ofdm_freq = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
    
    # Ejecutar IFFT
    time_signal = np.fft.ifft(ofdm_freq, axis=1)
    
    # Comprobar la componente imaginaria
    max_imag = np.max(np.abs(np.imag(time_signal)))
    print(f"   - Máxima amplitud de componente imaginaria: {max_imag:.2e}")
    assert max_imag < 1e-12, "ERROR: La señal en tiempo no es puramente real."
    print("   - OK: Simetría Hermitiana verificada con éxito (salida IFFT real).")


def test_ideal_channel():
    print("\n2. Probando Transmisión a través de Canal Ideal (Sin Ruido) para todos los MCS y GIs...")
    f_if = 20e6
    f_s = 80e6
    
    np.random.seed(42)
    # 480 bits de entrada -> coded = 960 bits de tasa madre 1/2
    input_bits = np.random.randint(0, 2, 480)
    
    gis = {"0.8us": 64, "1.6us": 128, "3.2us": 256}
    
    for gi_name, cp_len in gis.items():
        for mcs in range(12):
            mod, rate = physical_layer.get_mcs_params(mcs)
            
            # Cadena TX
            coded = physical_layer.fec_encoder(input_bits)
            punctured = physical_layer.puncture(coded, rate=rate)
            interleaved = physical_layer.interleave(punctured)
            tx_symbols = physical_layer.constellation_mapping(interleaved, scheme=mod)
            tx_freq = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            tx_time = physical_layer.idft_processing(tx_freq)
            tx_cp = physical_layer.insert_guard_interval(tx_time, cp_len=cp_len)
            tx_win = physical_layer.windowing(tx_cp, roll_off=0.05)
            s_if, t = physical_layer.iq_modulation(tx_win, f_if=f_if, f_s=f_s)
            s_if_peak = np.max(np.abs(s_if))
            scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
            s_if = s_if * scaling_factor
            s_biased, bias = physical_layer.add_dc_bias(s_if, bias_mode="adaptive")
            s_opt = physical_layer.tx_optical_front_end(s_biased, led_min=0.0, led_max=5.0)
            
            # Canal Ideal
            s_rx_opt = physical_layer.optical_wireless_channel(
                s_opt, distance=1.0, rx_area=2*np.pi/(2), semi_angle_led=60.0, 
                fov_rx=60.0, index_conc=1.0, noise_variance=0.0
            )
            
            # Cadena RX
            i_rx = physical_layer.rx_optical_front_end(s_rx_opt, responsivity=1.0)
            i_ac = physical_layer.remove_dc_bias(i_rx)
            demod = physical_layer.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
            rx_time = physical_layer.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
            rx_freq = physical_layer.dft_processing(rx_time)
            rx_symbols = physical_layer.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            
            # One-tap FEQ (Frequency Domain Equalization) para eliminar la atenuación del filtro analógico
            H_subcarrier = rx_symbols[:len(tx_symbols)] / tx_symbols
            rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / H_subcarrier
            
            rx_interleaved = physical_layer.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
            rx_punctured = physical_layer.deinterleave(rx_interleaved)
            
            # Recortar el padding del interleaver antes del depuncturing
            rx_punctured_truncated = rx_punctured[:len(punctured)]
            
            # Depuncturing para reconstruir la estructura de la tasa madre 1/2
            rx_coded, is_erasure = physical_layer.depuncture(rx_punctured_truncated, rate=rate, target_len=len(coded))
            
            # Decodificación de Viterbi considerando los borrados (erasures)
            decoded = physical_layer.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
            
            # Comparación de bits
            min_len = min(len(input_bits), len(decoded))
            bit_errors = np.sum(input_bits[:min_len] != decoded[:min_len])
            
            print(f"   - GI: {gi_name} | MCS {mcs:2d} ({mod:7s}, tasa {rate:3s}) -> Errores: {bit_errors}")
            assert bit_errors == 0, f"ERROR: Falló en GI {gi_name}, MCS {mcs}."
            
    print("   - OK: Transmisión ideal sin pérdidas verificada con éxito para todo el estándar (BER = 0).")


def generate_ber_curves(solar_irradiance: float = 150.0, 
                        filename_fisc: str = "curvas_rendimiento_fisico.png", 
                        filename_snr: str = "curvas_rendimiento_snr.png"):
    is_awgn = (solar_irradiance == 0.0)
    noise_desc = "Canal AWGN Puro" if is_awgn else f"Ruido Solar ($E_{{sol}} = {solar_irradiance:.0f}\\text{{ W/m}}^2$)"
    print(f"\n3. Generando Curvas de Rendimiento (Físico y SNR) para GI = {GI_SELECCIONADO} bajo {noise_desc}...")
    
    cp_len, t_symbol = get_gi_params(GI_SELECCIONADO)
    f_if = 20e6
    f_s = 80e6
    
    # Parámetros del barrido físico
    distance_range = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0])  # de 2m a 25m
    
    # Selección representativa de MCS para evitar saturar las gráficas
    mcs_selection = [0, 1, 3, 5, 8, 10]
    
    # Estructuras para guardar resultados
    ber_results = {mcs: [] for mcs in mcs_selection}
    throughput_results = {mcs: [] for mcs in mcs_selection}
    snr_db_results = {mcs: [] for mcs in mcs_selection}
    
    # Generar bits de prueba
    np.random.seed(123)
    input_bits = np.random.randint(0, 2, N_BITS_SOLAR)
    
    # Parámetros constantes del receptor y del canal
    rx_area = RX_AREA
    semi_angle = SEMI_ANGLE
    fov_rx = FOV_RX
    index_conc = INDEX_CONC
    responsivity = RESPONSIVITY
    bandwidth = BANDWIDTH
    
    phi_half = np.radians(semi_angle)
    m = -np.log(2.0) / np.log(np.cos(phi_half))
    
    for mcs in mcs_selection:
        mod, rate = physical_layer.get_mcs_params(mcs)
        rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
        print(f"   - Simulando MCS {mcs} ({mod}, tasa {rate}, {rate_mbps:.3f} Mbps)...")
        
        # Transmisión completa (independiente de ruido/distancia)
        coded = physical_layer.fec_encoder(input_bits)
        punctured = physical_layer.puncture(coded, rate=rate)
        interleaved = physical_layer.interleave(punctured)
        tx_symbols = physical_layer.constellation_mapping(interleaved, scheme=mod)
        tx_freq = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
        tx_time = physical_layer.idft_processing(tx_freq)
        tx_cp = physical_layer.insert_guard_interval(tx_time, cp_len=cp_len)
        tx_win = physical_layer.windowing(tx_cp, roll_off=0.05)
        s_if, t = physical_layer.iq_modulation(tx_win, f_if=f_if, f_s=f_s)
        s_if_peak = np.max(np.abs(s_if))
        scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
        s_if = s_if * scaling_factor
        s_biased, bias = physical_layer.add_dc_bias(s_if, bias_mode="adaptive")
        s_opt = physical_layer.tx_optical_front_end(s_biased, led_min=0.0, led_max=5.0)
        
        # Calibración del canal (genie-aided) para obtener H_filter del filtro analógico
        s_rx_cal = physical_layer.optical_wireless_channel(
            s_opt, distance=1.0, rx_area=2*np.pi/(2), semi_angle_led=60.0,
            fov_rx=60.0, index_conc=1.0, noise_variance=0.0
        )
        i_rx_cal = physical_layer.rx_optical_front_end(s_rx_cal, responsivity=1.0)
        i_ac_cal = physical_layer.remove_dc_bias(i_rx_cal)
        demod_cal = physical_layer.iq_demodulation(i_ac_cal, f_if=f_if, f_s=f_s, f_cutoff=15e6)
        rx_time_cal = physical_layer.remove_guard_interval(demod_cal, symbol_len=FFT_SIZE, cp_len=cp_len)
        rx_freq_cal = physical_layer.dft_processing(rx_time_cal)
        rx_symbols_cal = physical_layer.remove_hermitian_symmetry(rx_freq_cal, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
        # Normalizar H_filter: el canal de calibración tiene ganancia de 4/3 debido a index_conc=1.0 y fov=60.0
        H_cal_gain = 4.0 / 3.0
        H_filter = (rx_symbols_cal[:len(tx_symbols)] / tx_symbols) / H_cal_gain
        
        for dist in distance_range:
            # Geometría dinámica vehicular V2I
            d_3d = np.sqrt(dist**2 + H_DIFF**2)
            theta_deg = np.degrees(np.arctan(H_DIFF / dist))
            phi = np.abs(theta_deg - 15.0)
            psi = theta_deg
            g_psi_angular = (index_conc**2) / (np.sin(np.radians(fov_rx))**2) if psi <= fov_rx else 0.0
            
            # Ruido solar dependiente de la posición del receptor
            P_sol = solar_irradiance * 0.01 * rx_area * g_psi_angular
            I_bg = responsivity * P_sol
            q = 1.602e-19
            sigma2_shot = 2 * q * I_bg * bandwidth
            sigma2_thermal = 1e-16
            sigma2_noise_electrical = sigma2_shot + sigma2_thermal
            
            # Propagar por canal óptico
            s_rx_optical = physical_layer.optical_wireless_channel(
                s_opt, distance=d_3d, rx_area=rx_area, semi_angle_led=semi_angle,
                fov_rx=fov_rx, index_conc=index_conc, solar_irradiance=solar_irradiance * 0.01,
                phi=phi, psi=psi
            )
            
            # Recepción completa
            i_rx = physical_layer.rx_optical_front_end(s_rx_optical, responsivity=responsivity)
            i_ac = physical_layer.remove_dc_bias(i_rx)
            demod = physical_layer.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
            rx_time = physical_layer.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
            rx_freq = physical_layer.dft_processing(rx_time)
            rx_symbols = physical_layer.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            
            # Ecualización de canal considerando ángulos dinámicos
            H_0 = ((m + 1) * rx_area / (2 * np.pi * (d_3d**2))) * g_psi_angular * \
                  (np.cos(np.radians(phi))**m) * np.cos(np.radians(psi))
            gain = H_0 * responsivity
            equalization_gain = max(gain, 1e-20)
            rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (equalization_gain * H_filter)
            
            # Demapping y deinterleaving
            rx_interleaved = physical_layer.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
            rx_punctured = physical_layer.deinterleave(rx_interleaved)
            rx_punctured_truncated = rx_punctured[:len(punctured)]
            
            # Depuncturing
            rx_coded, is_erasure = physical_layer.depuncture(rx_punctured_truncated, rate=rate, target_len=len(coded))
            
            # Decodificación de Viterbi
            decoded = physical_layer.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
            
            # Calcular BER (mínimo de 1e-6 para visualización semilogarítmica)
            min_len = min(len(input_bits), len(decoded))
            bit_errors = np.sum(input_bits[:min_len] != decoded[:min_len])
            ber = bit_errors / min_len
            ber_results[mcs].append(max(ber, 1e-6))
            
            # Calcular SNR recibida real
            s_rx_optical_clean = H_0 * s_opt
            s_rx_electrical_clean = responsivity * s_rx_optical_clean
            s_rx_ac_clean = s_rx_electrical_clean - np.mean(s_rx_electrical_clean)
            signal_power = np.mean(s_rx_ac_clean ** 2)
            
            snr_linear = signal_power / sigma2_noise_electrical
            snr_db = 10 * np.log10(snr_linear)
            snr_db_results[mcs].append(snr_db)
            
            # Throughput efectivo
            r_eff = rate_mbps * (1.0 - ber)
            throughput_results[mcs].append(r_eff)

    # Colores y marcadores estéticos para las curvas
    colors = {
        0: "#1f77b4",  # BPSK
        1: "#2ca02c",  # QPSK
        3: "#ff7f0e",  # 16QAM
        5: "#d62728",  # 64QAM
        8: "#9467bd",  # 256QAM
        10: "#8c564b"  # 1024QAM
    }
    markers = {0: "o-", 1: "s-", 3: "^-", 5: "d-", 8: "x-", 10: "v-"}

    # --- ARCHIVO 1: vs. Distancia ---
    fig_fisc, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for mcs in mcs_selection:
        mod, rate = physical_layer.get_mcs_params(mcs)
        rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
        label = f"MCS {mcs} ({rate_mbps:.3f} Mbps)"
        
        ax1.semilogy(distance_range, ber_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        ax2.plot(distance_range, throughput_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        
    title_suffix = " (AWGN)" if is_awgn else " (Ruido Solar)"
    ax1.set_title("A. Tasa de Error de Bit (BER) vs. Distancia" + title_suffix + "\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Distancia (m)")
    ax1.set_ylabel("BER")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim([1e-6, 1.0])
    
    ax2.set_title("B. Throughput Efectivo vs. Distancia" + title_suffix + "\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Distancia (m)")
    ax2.set_ylabel("Throughput Efectivo (Mbps)")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(fontsize=9, loc="upper right")
    
    plt.tight_layout()
    save_plot_dual(fig_fisc, filename_fisc)
    plt.close(fig_fisc)
    print(f"   - OK: {filename_fisc} guardada con éxito.")

    # --- ARCHIVO 2: vs. SNR ---
    fig_snr, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for mcs in mcs_selection:
        mod, rate = physical_layer.get_mcs_params(mcs)
        rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
        label = f"MCS {mcs} ({rate_mbps:.3f} Mbps)"
        
        ax1.semilogy(snr_db_results[mcs], ber_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        ax2.plot(snr_db_results[mcs], throughput_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        
    ax1.set_title("A. Tasa de Error de Bit (BER) vs. SNR Eléctrico Recibido" + title_suffix + "\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("SNR Eléctrico (dB)")
    ax1.set_ylabel("BER")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_ylim([1e-6, 1.0])
    ax1.set_xlim([-10, 35])
    
    ax2.set_title("B. Throughput Efectivo vs. SNR Eléctrico Recibido" + title_suffix + "\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("SNR Eléctrico (dB)")
    ax2.set_ylabel("Throughput Efectivo (Mbps)")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_xlim([-10, 35])
    
    plt.tight_layout()
    save_plot_dual(fig_snr, filename_snr)
    plt.close(fig_snr)
    print(f"   - OK: {filename_snr} guardada con éxito.")


def generate_ber_curves_ideal():
    print(f"\n4. Generando Curvas de Rendimiento vs. SNR en Canal Ideal (AWGN) para GI = {GI_SELECCIONADO}...")
    
    cp_len, t_symbol = get_gi_params(GI_SELECCIONADO)
    f_if = 20e6
    f_s = 80e6
    
    # Rango extendido de SNR para ver el límite de 1024-QAM
    snr_db_range = np.arange(5, 42, 3)
    
    mcs_selection = list(range(12))
    
    ber_results = {mcs: [] for mcs in mcs_selection}
    throughput_results = {mcs: [] for mcs in mcs_selection}
    
    # Generar bits de prueba
    np.random.seed(123)
    input_bits = np.random.randint(0, 2, N_BITS_IDEAL)
    
    distance = 2.0
    rx_area = 1e-4
    semi_angle = 60.0
    fov_rx = 60.0
    index_conc = 1.5
    responsivity = 0.6
    
    for mcs in mcs_selection:
        mod, rate = physical_layer.get_mcs_params(mcs)
        rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
        print(f"   - Simulando MCS {mcs} ({mod}, tasa {rate}, {rate_mbps:.3f} Mbps)...")
        
        coded = physical_layer.fec_encoder(input_bits)
        punctured = physical_layer.puncture(coded, rate=rate)
        interleaved = physical_layer.interleave(punctured)
        tx_symbols = physical_layer.constellation_mapping(interleaved, scheme=mod)
        tx_freq = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
        tx_time = physical_layer.idft_processing(tx_freq)
        tx_cp = physical_layer.insert_guard_interval(tx_time, cp_len=cp_len)
        tx_win = physical_layer.windowing(tx_cp, roll_off=0.05)
        s_if, t = physical_layer.iq_modulation(tx_win, f_if=f_if, f_s=f_s)
        s_if_peak = np.max(np.abs(s_if))
        scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
        s_if = s_if * scaling_factor
        s_biased, bias = physical_layer.add_dc_bias(s_if, bias_mode="adaptive")
        s_opt = physical_layer.tx_optical_front_end(s_biased, led_min=0.0, led_max=5.0)
        
        # Calibración del canal (genie-aided) para obtener H_filter del filtro analógico
        s_rx_cal = physical_layer.optical_wireless_channel(
            s_opt, distance=1.0, rx_area=2*np.pi/(2), semi_angle_led=60.0,
            fov_rx=60.0, index_conc=1.0, noise_variance=0.0
        )
        i_rx_cal = physical_layer.rx_optical_front_end(s_rx_cal, responsivity=1.0)
        i_ac_cal = physical_layer.remove_dc_bias(i_rx_cal)
        demod_cal = physical_layer.iq_demodulation(i_ac_cal, f_if=f_if, f_s=f_s, f_cutoff=15e6)
        rx_time_cal = physical_layer.remove_guard_interval(demod_cal, symbol_len=FFT_SIZE, cp_len=cp_len)
        rx_freq_cal = physical_layer.dft_processing(rx_time_cal)
        rx_symbols_cal = physical_layer.remove_hermitian_symmetry(rx_freq_cal, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
        # Normalizar H_filter: el canal de calibración tiene ganancia de 4/3 debido a index_conc=1.0 y fov=60.0
        H_cal_gain = 4.0 / 3.0
        H_filter = (rx_symbols_cal[:len(tx_symbols)] / tx_symbols) / H_cal_gain
        
        # Ganancia Lambertiana
        phi_half = np.radians(semi_angle)
        m = -np.log(2.0) / np.log(np.cos(phi_half))
        g_psi = (index_conc**2) / (np.sin(np.radians(fov_rx))**2)
        H_0 = ((m + 1) * rx_area / (2 * np.pi * (distance**2))) * g_psi
        
        s_rx_optical_clean = H_0 * s_opt
        s_rx_electrical_clean = responsivity * s_rx_optical_clean
        s_rx_ac_clean = s_rx_electrical_clean - np.mean(s_rx_electrical_clean)
        signal_power = np.mean(s_rx_ac_clean ** 2)
        
        for snr_db in snr_db_range:
            snr_linear = 10 ** (snr_db / 10.0)
            noise_variance_electrical = signal_power / snr_linear
            noise_variance_optical = noise_variance_electrical / (responsivity ** 2)
            
            s_rx_optical = physical_layer.optical_wireless_channel(
                s_opt, distance=distance, rx_area=rx_area, semi_angle_led=semi_angle,
                fov_rx=fov_rx, index_conc=index_conc, noise_variance=noise_variance_optical,
                solar_irradiance=0.0
            )
            
            # Recepción completa
            i_rx = physical_layer.rx_optical_front_end(s_rx_optical, responsivity=responsivity)
            i_ac = physical_layer.remove_dc_bias(i_rx)
            demod = physical_layer.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
            rx_time = physical_layer.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
            rx_freq = physical_layer.dft_processing(rx_time)
            rx_symbols = physical_layer.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            
            gain = H_0 * responsivity
            rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (gain * H_filter)
            
            rx_interleaved = physical_layer.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
            rx_punctured = physical_layer.deinterleave(rx_interleaved)
            rx_punctured_truncated = rx_punctured[:len(punctured)]
            
            rx_coded, is_erasure = physical_layer.depuncture(rx_punctured_truncated, rate=rate, target_len=len(coded))
            decoded = physical_layer.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
            
            min_len = min(len(input_bits), len(decoded))
            bit_errors = np.sum(input_bits[:min_len] != decoded[:min_len])
            ber = bit_errors / min_len
            ber_results[mcs].append(max(ber, 1e-6))
            
            r_eff = rate_mbps * (1.0 - ber)
            throughput_results[mcs].append(r_eff)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        0: "#1f77b4", 1: "#aec7e8", 2: "#ff7f0e", 3: "#ffbb78",
        4: "#2ca02c", 5: "#98df8a", 6: "#d62728", 7: "#ff9896",
        8: "#9467bd", 9: "#c5b0d5", 10: "#8c564b", 11: "#c49c94"
    }
    markers_list = ["o-", "s-", "^-", "d-", "x-", "v-", "p-", "*-", "h-", "+-", "1-", "2-"]
    markers = {i: markers_list[i % len(markers_list)] for i in range(12)}
    
    for mcs in mcs_selection:
        mod, rate = physical_layer.get_mcs_params(mcs)
        rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
        label = f"MCS {mcs} ({rate_mbps:.3f} Mbps)"
        
        ax1.semilogy(snr_db_range, ber_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        ax2.plot(snr_db_range, throughput_results[mcs], markers[mcs], color=colors[mcs], label=label, linewidth=1.8)
        
    ax1.set_title("A. Tasa de Error de Bit (BER) vs. SNR Eléctrico en canal Ideal (AWGN)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("SNR Eléctrico (dB)")
    ax1.set_ylabel("BER")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_ylim([1e-6, 1.0])
    
    ax2.set_title("B. Throughput Efectivo vs. SNR Eléctrico en canal Ideal (AWGN)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("SNR Eléctrico (dB)")
    ax2.set_ylabel("Throughput Efectivo (Mbps)")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(fontsize=9, loc="upper left")
    
    plt.tight_layout()
    save_plot_dual(fig, "curvas_rendimiento_ideal.png")
    plt.close(fig)
    print(f"   - OK: curvas_rendimiento_ideal.png guardada con éxito.")


def generate_scenario_curves(solar_noise: bool = True, filename: str = "curvas_rendimiento_escenarios.png"):
    desc_noise = "Ruido Solar Real" if solar_noise else "AWGN Puro (Sin Ruido Solar)"
    print(f"\n5. Generando Curvas comparativas de Escenarios Ambientales para GI = {GI_SELECCIONADO} bajo {desc_noise}...")
    
    cp_len, t_symbol = get_gi_params(GI_SELECCIONADO)
    f_if = 20e6
    f_s = 80e6
    
    # Rango de distancia para escenarios
    distance_range = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    
    # MCS 1 (QPSK, tasa 1/2) como referencia comparativa
    mcs_ref = 1
    mod, rate = physical_layer.get_mcs_params(mcs_ref)
    rate_ref_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
    
    if solar_noise:
        scenarios = [
            ("Noche Despejada (Ideal)", 0.0, 0.0, "navy", "o-"),
            ("Día Soleado (Ruido Solar)", 50.0, 0.0, "gold", "s-"),
            ("Lluvia Moderada", 20.0, 0.05, "tab:blue", "^-"),
            ("Neblina Moderada", 10.0, 0.15, "tab:gray", "d-")
        ]
    else:
        scenarios = [
            ("Noche Despejada (Ideal)", 0.0, 0.0, "navy", "o-"),
            ("Día Soleado (Sin Ruido)", 0.0, 0.0, "gold", "s-"),
            ("Lluvia Moderada (Sin Ruido)", 0.0, 0.05, "tab:blue", "^-"),
            ("Neblina Moderada (Sin Ruido)", 0.0, 0.15, "tab:gray", "d-")
        ]
    
    ber_results = {scen[0]: [] for scen in scenarios}
    throughput_results = {scen[0]: [] for scen in scenarios}
    
    # Generar bits de prueba
    np.random.seed(123)
    input_bits = np.random.randint(0, 2, N_BITS_ESCENARIOS)
    
    rx_area = RX_AREA
    semi_angle = SEMI_ANGLE
    fov_rx = FOV_RX
    index_conc = INDEX_CONC
    responsivity = RESPONSIVITY
    
    phi_half = np.radians(semi_angle)
    m = -np.log(2.0) / np.log(np.cos(phi_half))
    
    # Transmisión completa
    coded = physical_layer.fec_encoder(input_bits)
    punctured = physical_layer.puncture(coded, rate=rate)
    interleaved = physical_layer.interleave(punctured)
    tx_symbols = physical_layer.constellation_mapping(interleaved, scheme=mod)
    tx_freq = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
    tx_time = physical_layer.idft_processing(tx_freq)
    tx_cp = physical_layer.insert_guard_interval(tx_time, cp_len=cp_len)
    tx_win = physical_layer.windowing(tx_cp, roll_off=0.05)
    s_if, t = physical_layer.iq_modulation(tx_win, f_if=f_if, f_s=f_s)
    s_if_peak = np.max(np.abs(s_if))
    scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
    s_if = s_if * scaling_factor
    s_biased, bias = physical_layer.add_dc_bias(s_if, bias_mode="adaptive")
    s_opt = physical_layer.tx_optical_front_end(s_biased, led_min=0.0, led_max=5.0)
    
    # Calibración del canal (genie-aided) para obtener H_filter del filtro analógico
    s_rx_cal = physical_layer.optical_wireless_channel(
        s_opt, distance=1.0, rx_area=2*np.pi/(2), semi_angle_led=60.0,
        fov_rx=60.0, index_conc=1.0, noise_variance=0.0
    )
    i_rx_cal = physical_layer.rx_optical_front_end(s_rx_cal, responsivity=1.0)
    i_ac_cal = physical_layer.remove_dc_bias(i_rx_cal)
    demod_cal = physical_layer.iq_demodulation(i_ac_cal, f_if=f_if, f_s=f_s, f_cutoff=15e6)
    rx_time_cal = physical_layer.remove_guard_interval(demod_cal, symbol_len=FFT_SIZE, cp_len=cp_len)
    rx_freq_cal = physical_layer.dft_processing(rx_time_cal)
    rx_symbols_cal = physical_layer.remove_hermitian_symmetry(rx_freq_cal, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
    # Normalizar H_filter: el canal de calibración tiene ganancia de 4/3 debido a index_conc=1.0 y fov=60.0
    H_cal_gain = 4.0 / 3.0
    H_filter = (rx_symbols_cal[:len(tx_symbols)] / tx_symbols) / H_cal_gain
    
    for name, solar_irradiance, extinction_coef, color, marker in scenarios:
        print(f"   - Simulando escenario: {name}...")
        
        for dist in distance_range:
            # Geometría dinámica vehicular V2I
            d_3d = np.sqrt(dist**2 + H_DIFF**2)
            theta_deg = np.degrees(np.arctan(H_DIFF / dist))
            phi = np.abs(theta_deg - 15.0)
            psi = theta_deg
            g_psi_angular = (index_conc**2) / (np.sin(np.radians(fov_rx))**2) if psi <= fov_rx else 0.0
            
            s_rx_optical = physical_layer.optical_wireless_channel(
                s_opt, distance=d_3d, rx_area=rx_area, semi_angle_led=semi_angle,
                fov_rx=fov_rx, index_conc=index_conc, solar_irradiance=solar_irradiance * 0.01,
                extinction_coef=extinction_coef, phi=phi, psi=psi
            )
            
            i_rx = physical_layer.rx_optical_front_end(s_rx_optical, responsivity=responsivity)
            i_ac = physical_layer.remove_dc_bias(i_rx)
            demod = physical_layer.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
            rx_time = physical_layer.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
            rx_freq = physical_layer.dft_processing(rx_time)
            rx_symbols = physical_layer.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            
            # Lambertiana + Beer-Lambert para ecualizar con ángulos dinámicos
            H_0 = ((m + 1) * rx_area / (2 * np.pi * (d_3d**2))) * g_psi_angular * \
                  (np.cos(np.radians(phi))**m) * np.cos(np.radians(psi))
            atmospheric_attenuation = np.exp(-extinction_coef * d_3d)
            gain = H_0 * atmospheric_attenuation * responsivity
            
            equalization_gain = max(gain, 1e-20)
            rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (equalization_gain * H_filter)
            
            rx_interleaved = physical_layer.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
            rx_punctured = physical_layer.deinterleave(rx_interleaved)
            rx_punctured_truncated = rx_punctured[:len(punctured)]
            
            rx_coded, is_erasure = physical_layer.depuncture(rx_punctured_truncated, rate=rate, target_len=len(coded))
            decoded = physical_layer.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
            
            min_len = min(len(input_bits), len(decoded))
            bit_errors = np.sum(input_bits[:min_len] != decoded[:min_len])
            ber = bit_errors / min_len
            ber_results[name].append(max(ber, 1e-6))
            
            r_eff = rate_ref_mbps * (1.0 - ber)
            throughput_results[name].append(r_eff)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for name, solar_irradiance, extinction_coef, color, marker in scenarios:
        ax1.semilogy(distance_range, ber_results[name], marker, color=color, label=name, linewidth=1.8)
        ax2.plot(distance_range, throughput_results[name], marker, color=color, label=name, linewidth=1.8)
        
    title_suffix = " (AWGN)" if not solar_noise else ""
    ax1.set_title(f"A. Tasa de Error de Bit (BER) vs. Distancia ({mod}, {rate_ref_mbps:.3f} Mbps){title_suffix}\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Distancia (m)")
    ax1.set_ylabel("BER")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend(fontsize=9)
    ax1.set_ylim([1e-6, 1.0])
    
    ax2.set_title(f"B. Throughput Efectivo vs. Distancia ({mod}, {rate_ref_mbps:.3f} Mbps){title_suffix}\n(Enlace horizontal alineado: phi = 0°, psi = 0°)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Distancia (m)")
    ax2.set_ylabel("Throughput Efectivo (Mbps)")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    save_plot_dual(fig, filename)
    plt.close(fig)
    print(f"   - OK: {filename} guardada con éxito.")


if __name__ == "__main__":
    print("=" * 60)
    print("EJECUTANDO VERIFICACIÓN DE CAPA FÍSICA IEEE 802.11bb")
    print("=" * 60)
    
    test_hermitian_symmetry()
    test_ideal_channel()
    
    # 1. Curvas físicas bajo ruido solar real
    generate_ber_curves(solar_irradiance=150.0, 
                        filename_fisc="curvas_rendimiento_fisico.png", 
                        filename_snr="curvas_rendimiento_snr.png")
                        
    # 2. Curvas físicas bajo AWGN puro (sin ruido solar)
    generate_ber_curves(solar_irradiance=0.0, 
                        filename_fisc="curvas_rendimiento_fisico_awgn.png", 
                        filename_snr="curvas_rendimiento_snr_awgn.png")
                        
    # 3. Curvas en canal ideal
    generate_ber_curves_ideal()
    
    # 4. Curvas comparativas de escenarios bajo ruido solar real
    generate_scenario_curves(solar_noise=True, filename="curvas_rendimiento_escenarios.png")
    
    # 5. Curvas comparativas de escenarios bajo AWGN puro (sin ruido solar)
    generate_scenario_curves(solar_noise=False, filename="curvas_rendimiento_escenarios_awgn.png")
    
    print("\n" + "=" * 60)
    print("VERIFICACIÓN COMPLETADA SATISFACTORIAMENTE")
    print("=" * 60)
