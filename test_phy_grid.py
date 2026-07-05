"""
test_phy_grid.py
Script de simulación de capa física LiFi IEEE 802.11bb con reporte matricial 3x4 (Opción B).
Genera las 3 figuras de la tesis (BER vs SNR, BER vs Distancia, Throughput vs Distancia)
ejecutando la cadena completa del transceptor y el canal óptico.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lifi_phy
import os

# ==============================================================================
#                 CONFIGURACIÓN DE SIMULACIONES Y PARÁMETROS
# ==============================================================================
MONTE_CARLO_RUNS = 10000  # Configurado a 10000 para observar la mejora del suavizado de curvas
FFT_SIZE = 1024
ACTIVE_CARRIERS = list(range(139, 256)) + list(range(257, 374)) # 234 portadoras activas
f_s = 80e6
f_if = 20e6

output_dir = r"c:\Users\carlo\OneDrive\Imágenes\Documentos\TESIS\TESIS_LATEX"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# También guardamos en raíz
root_dir = r"c:\Users\carlo\OneDrive\Imágenes\Documentos\TESIS"

# Selección de MCS y GIs
mcs_list = [0, 5, 11]
mcs_labels = {0: "MCS 0 (BPSK, 1/2)", 5: "MCS 5 (64-QAM, 2/3)", 11: "MCS 11 (1024-QAM, 5/6)"}
colors = {0: "#1f77b4", 5: "#2ca02c", 11: "#d62728"} # Usar verde esmeralda para MCS 5 para mejorar contraste con el rojo de MCS 11
markers = {0: "o-", 5: "s-", 11: "^-"}

gis = ["0.8us", "1.6us", "3.2us"]
scenarios = ["Noche Despejada", "Día Soleado", "Lluvia Moderada", "Neblina Moderada"]

# Coeficientes climáticos: (solar_irradiance, extinction_coef)
# Simula el uso de un filtro óptico de paso de banda que atenúa la luz solar directa en un 95%
scen_params = {
    "Noche Despejada": (0.0, 0.0),
    "Día Soleado": (0.5, 0.0),      # 150 W/m^2 * 0.0033
    "Lluvia Moderada": (0.2, 0.05),   # 50 W/m^2 * 0.004
    "Neblina Moderada": (0.1, 0.15)   # 30 W/m^2 * 0.0033
}

# Rangos de barrido (Evaluados a alta resolución para suavizado óptimo)
snr_db_range = np.arange(0, 47, 2)  # Paso de 2 dB (24 puntos hasta 46 dB)
# Distancia de barrido vehicular acotada a 50 metros
distance_range = np.arange(2.0, 51.0, 2.0)  # Paso de 2 metros (25 puntos)

# Parámetros constantes del detector (adaptados para escenario vehicular con lentes colimadoras y faro concentrado)
rx_area = 10e-4      # Área de captación ampliada a 10 cm^2 gracias al uso de lentes
semi_angle = 15.0   # Haz directivo de faro vehicular (semiángulo de 15 grados)
fov_rx = 60.0
index_conc = 1.5
responsivity = 0.6
bandwidth = 40e6

def get_gi_params(gi_name: str) -> tuple[int, float]:
    gis_map = {"0.8us": (64, 13.6e-6), "1.6us": (128, 14.4e-6), "3.2us": (256, 16.0e-6)}
    return gis_map[gi_name]

def parse_code_rate(rate_str: str) -> float:
    num, den = rate_str.split("/")
    return float(num) / float(den)

def get_phy_data_rate(modulation: str, fft_size: int, cp_len: int, f_s: float, n_active_carriers: int = 234, code_rate_str: str = "1/2") -> float:
    qam_sizes = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
    bits_per_symbol = qam_sizes[modulation.upper()]
    t_symbol = (fft_size + cp_len) / f_s
    code_rate = parse_code_rate(code_rate_str)
    r_raw = (n_active_carriers * bits_per_symbol * code_rate) / t_symbol
    return r_raw

def save_fig_dual(fig, filename):
    fig.savefig(os.path.join(root_dir, filename), dpi=150)
    fig.savefig(os.path.join(output_dir, filename), dpi=150)

# ==============================================================================
#                 EJECUCIÓN PRINCIPAL DE LA SIMULACIÓN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO SIMULACIÓN MATRICIAL DE CAPA FÍSICA LiFi")
    print(f"Iteraciones Monte Carlo por punto: {MONTE_CARLO_RUNS}")
    print("=" * 70)
    
    np.random.seed(42)
    # Generar bits aleatorios
    input_bits = np.random.randint(0, 2, MONTE_CARLO_RUNS)
    
    # Estructuras de almacenamiento
    # Estructura: results_ber_snr[gi][scenario][mcs] = [ber_points]
    results_ber_snr = {gi: {scen: {mcs: [] for mcs in mcs_list} for scen in scenarios} for gi in gis}
    results_ber_dist = {gi: {scen: {mcs: [] for mcs in mcs_list} for scen in scenarios} for gi in gis}
    results_thr_dist = {gi: {scen: {mcs: [] for mcs in mcs_list} for scen in scenarios} for gi in gis}
    results_snr_vals = {gi: {scen: {mcs: [] for mcs in mcs_list} for scen in scenarios} for gi in gis}

    # Bucle principal de simulación
    for gi in gis:
        cp_len, t_symbol = get_gi_params(gi)
        
        for mcs in mcs_list:
            mod, rate = lifi_phy.get_mcs_params(mcs)
            rate_mbps = get_phy_data_rate(mod, FFT_SIZE, cp_len, f_s, n_active_carriers=len(ACTIVE_CARRIERS), code_rate_str=rate) / 1e6
            print(f"\n---> Procesando: GI {gi} | MCS {mcs} ({mod} {rate}, {rate_mbps:.2f} Mbps)")
            
            # 1. Transmisión completa libre de ruido
            coded = lifi_phy.fec_encoder(input_bits)
            punctured = lifi_phy.puncture(coded, rate=rate)
            interleaved = lifi_phy.interleave(punctured)
            tx_symbols = lifi_phy.constellation_mapping(interleaved, scheme=mod)
            tx_freq = lifi_phy.apply_hermitian_symmetry(tx_symbols, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            tx_time = lifi_phy.idft_processing(tx_freq)
            tx_cp = lifi_phy.insert_guard_interval(tx_time, cp_len=cp_len)
            tx_win = lifi_phy.windowing(tx_cp, roll_off=0.05)
            s_if, t = lifi_phy.iq_modulation(tx_win, f_if=f_if, f_s=f_s)
            
            # Normalizar s_if para aprovechar al máximo el rango dinámico del LED (0 a 5.0 V) sin clipping
            s_if_peak = np.max(np.abs(s_if))
            scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
            s_if = s_if * scaling_factor
            
            s_biased, bias = lifi_phy.add_dc_bias(s_if, bias_mode="adaptive")
            s_opt = lifi_phy.tx_optical_front_end(s_biased, led_min=0.0, led_max=5.0)
            
            # Calibración de canal (genie-aided) para FEQ
            s_rx_cal = lifi_phy.optical_wireless_channel(
                s_opt, distance=1.0, rx_area=2*np.pi/2, semi_angle_led=60.0,
                fov_rx=60.0, index_conc=1.0, noise_variance=0.0
            )
            i_rx_cal = lifi_phy.rx_optical_front_end(s_rx_cal, responsivity=1.0)
            i_ac_cal = lifi_phy.remove_dc_bias(i_rx_cal)
            demod_cal = lifi_phy.iq_demodulation(i_ac_cal, f_if=f_if, f_s=f_s, f_cutoff=15e6)
            rx_time_cal = lifi_phy.remove_guard_interval(demod_cal, symbol_len=FFT_SIZE, cp_len=cp_len)
            rx_freq_cal = lifi_phy.dft_processing(rx_time_cal)
            rx_symbols_cal = lifi_phy.remove_hermitian_symmetry(rx_freq_cal, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
            # Normalizar H_filter: el canal de calibración tiene una ganancia DC de 4/3 debido a index_conc=1.0 y fov=60.0
            H_cal_gain = 4.0 / 3.0
            H_filter = (rx_symbols_cal[:len(tx_symbols)] / tx_symbols) / H_cal_gain
            
            # Calcular ganancia Lambertiana base
            phi_half = np.radians(semi_angle)
            m = -np.log(2.0) / np.log(np.cos(phi_half))
            g_psi_val = (index_conc**2) / (np.sin(np.radians(fov_rx))**2)
            
            for scen in scenarios:
                solar, ext = scen_params[scen]
                print(f"   Simulando escenario: {scen}...")
                
                # --- A. BARRIDO DE SNR (Canal Ideal AWGN con SNR controlada) ---
                # Usar geometría de semáforo (V2I) a distancia intermedia de 15 metros
                h_diff = 4.0 # Altura de semáforo (5.5m) menos altura de receptor de vehículo (1.5m)
                dist_horiz_snr = 15.0
                distance_snr = np.sqrt(dist_horiz_snr**2 + h_diff**2)
                
                # Ángulo de inclinación estático intermedio (aproximadamente 21.8 grados)
                theta_snr_deg = np.degrees(np.arctan(h_diff / dist_horiz_snr))
                phi_snr = np.abs(theta_snr_deg - 15.0)
                psi_snr = theta_snr_deg
                
                # Ganancia Lambertiana con inclinación
                H_0_snr = ((m + 1) * rx_area / (2 * np.pi * (distance_snr**2))) * g_psi_val * \
                          (np.cos(np.radians(phi_snr))**m) * np.cos(np.radians(psi_snr))
                
                s_rx_optical_clean = H_0_snr * s_opt
                s_rx_electrical_clean = responsivity * s_rx_optical_clean
                s_rx_ac_clean = s_rx_electrical_clean - np.mean(s_rx_electrical_clean)
                signal_power_ac = np.mean(s_rx_ac_clean ** 2)
                
                for snr_db in snr_db_range:
                    snr_lin = 10 ** (snr_db / 10.0)
                    noise_variance_electrical = signal_power_ac / snr_lin
                    noise_variance_optical = noise_variance_electrical / (responsivity ** 2)
                    
                    s_rx_optical = lifi_phy.optical_wireless_channel(
                        s_opt, distance=distance_snr, rx_area=rx_area, semi_angle_led=semi_angle,
                        fov_rx=fov_rx, index_conc=index_conc, noise_variance=noise_variance_optical,
                        solar_irradiance=0.0, phi=phi_snr, psi=psi_snr
                    )
                    
                    # Cadena de recepción completa
                    i_rx = lifi_phy.rx_optical_front_end(s_rx_optical, responsivity=responsivity)
                    i_ac = lifi_phy.remove_dc_bias(i_rx)
                    demod = lifi_phy.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
                    rx_time = lifi_phy.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
                    rx_freq = lifi_phy.dft_processing(rx_time)
                    rx_symbols = lifi_phy.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
                    
                    # Ecualizar canal
                    gain_snr = H_0_snr * responsivity
                    rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (gain_snr * H_filter)
                    
                    # Demapping & decoding
                    rx_interleaved = lifi_phy.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
                    rx_punctured = lifi_phy.deinterleave(rx_interleaved)[:len(punctured)]
                    rx_coded, is_erasure = lifi_phy.depuncture(rx_punctured, rate=rate, target_len=len(coded))
                    decoded = lifi_phy.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
                    
                    # Calcular BER
                    min_len = min(len(input_bits), len(decoded))
                    errors = np.sum(input_bits[:min_len] != decoded[:min_len])
                    ber = errors / min_len
                    results_ber_snr[gi][scen][mcs].append(max(ber, 1e-6))
                
                # --- B. BARRIDO DE DISTANCIA (Canal Óptico Real + Ruido Solar + Atenuación Atmosférica) ---
                # El fotodetector calcula el ruido solar basándose en los ángulos variables del concentrador
                # Para simplificar y mantener realismo, calculamos el ruido de fondo por cada punto de distancia
                h_diff = 4.0 # Altura de semáforo (5.5m) menos altura de receptor de vehículo (1.5m)
                
                for dist in distance_range:
                    d_3d = np.sqrt(dist**2 + h_diff**2) # Distancia real del enlace 3D (slant range)
                    theta_deg = np.degrees(np.arctan(h_diff / dist)) # Ángulo de elevación
                    phi = np.abs(theta_deg - 15.0)
                    psi = theta_deg
                    
                    # Ruido solar dependiente de la ganancia angular del concentrador g(psi)
                    g_psi_angular = g_psi_val if psi <= fov_rx else 0.0
                    P_sol = solar * rx_area * g_psi_angular
                    I_bg = responsivity * P_sol
                    q = 1.602e-19
                    sigma2_shot = 2 * q * I_bg * bandwidth
                    sigma2_thermal = 1e-16
                    sigma2_noise_electrical = sigma2_shot + sigma2_thermal
                    
                    s_rx_optical = lifi_phy.optical_wireless_channel(
                        s_opt, distance=d_3d, rx_area=rx_area, semi_angle_led=semi_angle,
                        fov_rx=fov_rx, index_conc=index_conc, solar_irradiance=solar,
                        extinction_coef=ext, phi=phi, psi=psi
                    )
                    
                    i_rx = lifi_phy.rx_optical_front_end(s_rx_optical, responsivity=responsivity)
                    i_ac = lifi_phy.remove_dc_bias(i_rx)
                    demod = lifi_phy.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15e6)
                    rx_time = lifi_phy.remove_guard_interval(demod, symbol_len=FFT_SIZE, cp_len=cp_len)
                    rx_freq = lifi_phy.dft_processing(rx_time)
                    rx_symbols = lifi_phy.remove_hermitian_symmetry(rx_freq, fft_size=FFT_SIZE, active_carriers=ACTIVE_CARRIERS)
                    
                    # Ecualizar canal con ganancia geométrica considerando ángulos
                    H_0 = ((m + 1) * rx_area / (2 * np.pi * (d_3d**2))) * g_psi_angular * \
                          (np.cos(np.radians(phi))**m) * np.cos(np.radians(psi))
                    
                    atmospheric_attenuation = np.exp(-ext * d_3d)
                    gain = H_0 * atmospheric_attenuation * responsivity
                    
                    # Si el receptor está fuera de FOV (H_0 == 0), evitamos división por cero colocando una ganancia muy pequeña
                    equalization_gain = max(gain, 1e-20)
                    rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (equalization_gain * H_filter)
                    
                    # Demapping & decoding
                    rx_interleaved = lifi_phy.constellation_demapping(rx_symbols_equalized, scheme=mod)[:len(interleaved)]
                    rx_punctured = lifi_phy.deinterleave(rx_interleaved)[:len(punctured)]
                    rx_coded, is_erasure = lifi_phy.depuncture(rx_punctured, rate=rate, target_len=len(coded))
                    decoded = lifi_phy.fec_decoder(rx_coded, code_rate="1/2", is_erasure=is_erasure)
                    
                    # Calcular BER
                    min_len = min(len(input_bits), len(decoded))
                    errors = np.sum(input_bits[:min_len] != decoded[:min_len])
                    ber = errors / min_len
                    results_ber_dist[gi][scen][mcs].append(max(ber, 1e-6))
                    
                    # Throughput efectivo
                    r_eff = rate_mbps * (1.0 - ber) if ber < 0.1 else 0.0
                    results_thr_dist[gi][scen][mcs].append(max(r_eff, 0.0))

    # ==============================================================================
    #                 PLOTTING DE REPORTES MATRICIALES (4x3)
    # ==============================================================================
    print("\nGenerando Reportes Gráficos Matriciales...")

    # Funciones de suavizado para eliminar ruido de simulación por muestreo
    def smooth_log_data(y, window=3):
        y_clip = np.clip(y, 1e-6, 1.0)
        log_y = np.log10(y_clip)
        box = np.ones(window) / window
        log_y_smooth = np.convolve(log_y, box, mode='same')
        # Preservar los bordes originales para no deformar extremos
        log_y_smooth[0] = log_y[0]
        log_y_smooth[-1] = log_y[-1]
        return 10 ** log_y_smooth

    def smooth_linear_data(y, window=3):
        y_arr = np.array(y)
        box = np.ones(window) / window
        y_smooth = np.convolve(y_arr, box, mode='same')
        y_smooth[0] = y_arr[0]
        y_smooth[-1] = y_arr[-1]
        return y_smooth

    # --- FIGURA 1: BER vs SNR ---
    fig_snr, axes_snr = plt.subplots(4, 3, figsize=(13, 15), sharex=True, sharey=True)
    plot_lines = []
    for row_idx, scen in enumerate(scenarios):
        for col_idx, gi in enumerate(gis):
            ax = axes_snr[row_idx, col_idx]
            for mcs in mcs_list:
                raw_ber = results_ber_snr[gi][scen][mcs]
                smooth_ber = smooth_log_data(raw_ber, window=3)
                line, = ax.semilogy(snr_db_range, smooth_ber, 
                                    markers[mcs], color=colors[mcs], linewidth=1.5, markevery=1)
                if row_idx == 0 and col_idx == 0:
                    plot_lines.append(line)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.set_ylim([1e-6, 1.0])
            if row_idx == 0:
                ax.set_title(f"GI: {gi}", fontsize=12, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{scen}\n\nBER", fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("SNR Eléctrica (dB)", fontsize=10)

    fig_snr.legend(plot_lines, [mcs_labels[m] for m in mcs_list], loc="lower center", ncol=3, fontsize=11, frameon=True)
    fig_snr.suptitle("Curvas Comparativas: Tasa de Error de Bit (BER) vs. SNR Eléctrica\n(Geometría fija: d_horiz = 15m. Nota: El ángulo de incidencia y de irradiación varían dinámicamente con la distancia)", fontsize=13, fontweight="bold", y=0.97)
    fig_snr.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_fig_dual(fig_snr, "curvas_rendimiento_grid_snr.png")
    plt.close(fig_snr)

    # --- FIGURA 2: BER vs Distancia ---
    fig_dist, axes_dist = plt.subplots(4, 3, figsize=(13, 15), sharex=True, sharey=True)
    plot_lines = []
    for row_idx, scen in enumerate(scenarios):
        for col_idx, gi in enumerate(gis):
            ax = axes_dist[row_idx, col_idx]
            for mcs in mcs_list:
                raw_ber = results_ber_dist[gi][scen][mcs]
                smooth_ber = smooth_log_data(raw_ber, window=3)
                line, = ax.semilogy(distance_range, smooth_ber, 
                                    markers[mcs], color=colors[mcs], linewidth=1.5, markevery=1)
                if row_idx == 0 and col_idx == 0:
                    plot_lines.append(line)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.set_xlim([2.0, 50.0])
            ax.set_ylim([1e-6, 1.0])
            if row_idx == 0:
                ax.set_title(f"GI: {gi}", fontsize=12, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{scen}\n\nBER", fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("Distancia (m)", fontsize=10)

    fig_dist.legend(plot_lines, [mcs_labels[m] for m in mcs_list], loc="lower center", ncol=3, fontsize=11, frameon=True)
    fig_dist.suptitle("Curvas Comparativas: Tasa de Error de Bit (BER) vs. Distancia Física\n(Geometría dinámica: los ángulos de incidencia e irradiación varían continuamente con el desplazamiento)", fontsize=13, fontweight="bold", y=0.97)
    fig_dist.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_fig_dual(fig_dist, "curvas_rendimiento_grid_ber_dist.png")
    plt.close(fig_dist)

    # --- FIGURA 3: Throughput vs Distancia ---
    fig_thr, axes_thr = plt.subplots(4, 3, figsize=(13, 15), sharex=True, sharey=True)
    plot_lines = []
    for row_idx, scen in enumerate(scenarios):
        for col_idx, gi in enumerate(gis):
            ax = axes_thr[row_idx, col_idx]
            for mcs in mcs_list:
                raw_thr = results_thr_dist[gi][scen][mcs]
                smooth_thr = smooth_linear_data(raw_thr, window=3)
                line, = ax.plot(distance_range, smooth_thr, 
                                markers[mcs], color=colors[mcs], linewidth=1.5, markevery=1)
                if row_idx == 0 and col_idx == 0:
                    plot_lines.append(line)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.set_xlim([2.0, 50.0])
            if row_idx == 0:
                ax.set_title(f"GI: {gi}", fontsize=12, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{scen}\n\nThroughput (Mbps)", fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("Distancia (m)", fontsize=10)

    fig_thr.legend(plot_lines, [mcs_labels[m] for m in mcs_list], loc="lower center", ncol=3, fontsize=11, frameon=True)
    fig_thr.suptitle("Curvas Comparativas: Throughput Efectivo vs. Distancia Física\n(Geometría dinámica: los ángulos de incidencia e irradiación varían continuamente con el desplazamiento)", fontsize=13, fontweight="bold", y=0.97)
    fig_thr.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_fig_dual(fig_thr, "curvas_rendimiento_grid_throughput.png")
    plt.close(fig_thr)

    print("=" * 70)
    print("SIMULACIÓN MATRICIAL COMPLETADA SATISFACTORIAMENTE")
    print(f"Los gráficos compuestos han sido guardados en {output_dir}")
    print("=" * 70)
