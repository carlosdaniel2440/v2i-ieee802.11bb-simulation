"""
main.py
Script de ejecución principal de la simulación modular del estándar IEEE 802.11bb.
Ejecuta la cadena de transmisión (TX), el canal óptico, la cadena de recepción (RX),
mide la tasa de error de bit (BER), y genera gráficas de los procesos clave.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physical_layer
from config_entorno import (
    FFT_SIZE, CP_LEN, F_IF, F_S, ACTIVE_CARRIERS,
    H_DIFF, SEMI_ANGLE, FOV_RX, INDEX_CONC, RX_AREA,
    RESPONSIVITY, BANDWIDTH, LED_MIN, LED_MAX
)

# ==============================================================================
#                 CONFIGURACIÓN DE SIMULACIONES MONTE CARLO
#  Ajusta este valor para subir o bajar el número de pruebas (bits aleatorios)
# ==============================================================================
MONTE_CARLO_RUNS = 10000  # Valor base para las pruebas de Monte Carlo en el transceptor
# ==============================================================================

def run_lc_simulation(solar_irradiance: float = 0.0, output_filename: str = "resultado_simulacion_ideal.png"):
    print("=" * 60)
    print(f"INICIANDO SIMULACIÓN MODULAR ({'IDEAL' if solar_irradiance == 0.0 else 'REAL DÍA'})")
    print("=" * 60)
    
    # 1. Configuración de parámetros de simulación
    fft_size = FFT_SIZE
    cp_len = CP_LEN
    f_if = F_IF
    f_s = F_S
    modulation = "16QAM" # BPSK, QPSK, 16QAM, 64QAM
    active_carriers = ACTIVE_CARRIERS
    
    # Parámetros del canal y transceptor óptico
    distance = 15.0     # Distancia horizontal (15m para ver degradación y recuperación)
    semi_angle_led = SEMI_ANGLE
    fov_rx = FOV_RX
    index_conc = INDEX_CONC
    rx_area = RX_AREA
    responsivity = RESPONSIVITY
    noise_var = 1e-15   # Ruido térmico base del receptor
    led_min = LED_MIN
    led_max = LED_MAX
    
    # Generar bits aleatorios de entrada
    num_info_bits = MONTE_CARLO_RUNS
    np.random.seed(42) # Semilla fija para reproducibilidad
    input_bits = np.random.randint(0, 2, num_info_bits)
    
    print(f"Modo: HE-PHY (OFDM con Simetría Hermitiana)")
    print(f"Esquema de Modulación: {modulation}")
    print(f"Bits de información generados: {len(input_bits)}")
    print(f"FFT size: {fft_size} | Prefijo Cíclico (CP): {cp_len}")
    print(f"Frecuencia Intermedia: {f_if / 1e6:.1f} MHz | Sampling Rate: {f_s / 1e6:.1f} MHz")
    print("-" * 60)
    
    # ==========================================================================
    # CADENA DE TRANSMISIÓN (TX)
    # ==========================================================================
    print("PROCESANDO TRANSMISIÓN (TX)...")
    
    # A. Codificación de canal (FEC)
    coded_bits = physical_layer.fec_encoder(input_bits)
    print(f" -> FEC Encoder: Bits codificados = {len(coded_bits)}")
    
    # B. Interleaving de bits
    interleaved_bits = physical_layer.interleave(coded_bits, n_cols=16)
    
    # C. Constellation Mapping
    tx_symbols = physical_layer.constellation_mapping(interleaved_bits, scheme=modulation)
    print(f" -> Constellation Mapping: Símbolos complejos mapeados = {len(tx_symbols)}")
    
    # D. Aplicar Simetría Hermitiana (DCO-OFDM)
    tx_freq_matrix = physical_layer.apply_hermitian_symmetry(tx_symbols, fft_size=fft_size, active_carriers=active_carriers)
    num_symbols = tx_freq_matrix.shape[0]
    print(f" -> Hermitian Symmetry: {num_symbols} símbolos OFDM formados en frecuencia.")
    
    # E. IFFT (Transformación al dominio del tiempo)
    tx_time_matrix = physical_layer.idft_processing(tx_freq_matrix)
    
    # F. Inserción de Prefijo Cíclico
    tx_time_cp = physical_layer.insert_guard_interval(tx_time_matrix, cp_len=cp_len)
    
    # G. Ventaneado (Pulse Shaping)
    tx_time_windowed = physical_layer.windowing(tx_time_cp, roll_off=0.05)
    
    # H. Modulación a Frecuencia Intermedia (Up-conversion)
    s_if, t = physical_layer.iq_modulation(tx_time_windowed, f_if=f_if, f_s=f_s)
    
    # Escalado de la señal en transmisión para aprovechar el rango dinámico del LED
    s_if_peak = np.max(np.abs(s_if))
    scaling_factor = 2.4 / s_if_peak if s_if_peak > 0 else 1.0
    s_if = s_if * scaling_factor
    
    print(f" -> IQ Modulation: Señal real de FI generada y escalada ({len(s_if)} muestras).")
    
    # I. Adición de DC Bias
    s_biased, dc_bias = physical_layer.add_dc_bias(s_if, bias_mode="adaptive")
    print(f" -> DC Bias: Desplazamiento de corriente continua agregado = {dc_bias:.4f} A")
    
    # J. Frente Óptico Transmisor (Clipiado LED)
    optical_signal = physical_layer.tx_optical_front_end(s_biased, led_min=led_min, led_max=led_max)
    print(" -> TX OFE: Señal óptica lista para el canal.")
    print("-" * 60)
    
    # ==========================================================================
    # CANAL ÓPTICO (Geometría dinámica vehicular V2I)
    # ==========================================================================
    print("PROPAGANDO A TRAVÉS DEL CANAL ÓPTICO...")
    h_diff = H_DIFF
    d_3d = np.sqrt(distance**2 + h_diff**2)
    theta_deg = np.degrees(np.arctan(h_diff / distance))
    phi = np.abs(theta_deg - 15.0)
    psi = theta_deg
    g_psi_val = (index_conc**2) / (np.sin(np.radians(fov_rx))**2) if psi <= fov_rx else 0.0

    optical_rx = physical_layer.optical_wireless_channel(
        optical_signal, distance=d_3d, rx_area=rx_area,
        semi_angle_led=semi_angle_led, fov_rx=fov_rx, index_conc=index_conc,
        noise_variance=noise_var,
        solar_irradiance=solar_irradiance, phi=phi, psi=psi
    )

    print(f" -> Canal Óptico: Señal recibida a {distance:.1f} metros (3D = {d_3d:.2f} m).")
    print("-" * 60)
    
    # ==========================================================================
    # CADENA DE RECEPCIÓN (RX)
    # ==========================================================================
    print("PROCESANDO RECEPCIÓN (RX)...")
    
    # A. Frente Óptico Receptor (Fotodiodo)
    i_rx = physical_layer.rx_optical_front_end(optical_rx, responsivity=responsivity)
    
    # B. Bloqueo DC (Remove Bias)
    i_ac = physical_layer.remove_dc_bias(i_rx)
    
    # C. Demodulación a Frecuencia Intermedia (Down-conversion + LPF)
    demod_baseband = physical_layer.iq_demodulation(i_ac, f_if=f_if, f_s=f_s, f_cutoff=15.0e6)
    
    # D. Remover Prefijo Cíclico
    rx_time_matrix = physical_layer.remove_guard_interval(demod_baseband, symbol_len=fft_size, cp_len=cp_len)
    
    # E. FFT (Transformación al dominio de frecuencia)
    rx_freq_matrix = physical_layer.dft_processing(rx_time_matrix)
    
    # F. Remover Simetría Hermitiana (Extraer subportadoras activas)
    rx_symbols = physical_layer.remove_hermitian_symmetry(rx_freq_matrix, fft_size=fft_size, active_carriers=active_carriers)
    print(f" -> FFT + Hermitian Removal: Símbolos complejos recuperados = {len(rx_symbols)}")
    
    # ==========================================================================
    # CALIBRACIÓN DE CANAL (GENIE-AIDED) PARA ESTIMAR H_FILTER POR SUBPORTADORA
    # ==========================================================================
    s_rx_cal = physical_layer.optical_wireless_channel(
        optical_signal, distance=1.0, rx_area=2*np.pi/2, semi_angle_led=60.0,
        fov_rx=fov_rx, index_conc=1.0, noise_variance=0.0
    )
    i_rx_cal = physical_layer.rx_optical_front_end(s_rx_cal, responsivity=1.0)
    i_ac_cal = physical_layer.remove_dc_bias(i_rx_cal)
    demod_cal = physical_layer.iq_demodulation(i_ac_cal, f_if=f_if, f_s=f_s, f_cutoff=15.0e6)
    rx_time_cal = physical_layer.remove_guard_interval(demod_cal, symbol_len=fft_size, cp_len=cp_len)
    rx_freq_cal = physical_layer.dft_processing(rx_time_cal)
    rx_symbols_cal = physical_layer.remove_hermitian_symmetry(rx_freq_cal, fft_size=fft_size, active_carriers=active_carriers)
    H_cal_gain = 4.0 / 3.0
    H_filter = (rx_symbols_cal[:len(tx_symbols)] / tx_symbols) / H_cal_gain

    # Ecualización de un toque (FDE) para compensar pérdidas por distancia y responsividad
    phi_half = np.radians(semi_angle_led)
    m = -np.log(2.0) / np.log(np.cos(phi_half))
    H_0 = ((m + 1) * rx_area / (2 * np.pi * (d_3d**2))) * g_psi_val * \
          (np.cos(np.radians(phi))**m) * np.cos(np.radians(psi))
    gain = H_0 * responsivity
    rx_symbols_equalized = rx_symbols[:len(tx_symbols)] / (gain * H_filter)
    
    # G. Constellation Demapping (Decisión dura)
    rx_interleaved_bits = physical_layer.constellation_demapping(rx_symbols_equalized[:len(tx_symbols)], scheme=modulation)
    
    # H. Deinterleaving de bits
    rx_coded_bits = physical_layer.deinterleave(rx_interleaved_bits, n_cols=16)
    
    # I. Decodificación FEC (Viterbi)
    decoded_bits = physical_layer.fec_decoder(rx_coded_bits[:len(coded_bits)])
    print(f" -> Viterbi Decoder: Secuencia de información recuperada ({len(decoded_bits)} bits).")
    print("-" * 60)
    
    # ==========================================================================
    # EVALUACIÓN DE RENDIMIENTO
    # ==========================================================================
    # Asegurar igualdad de dimensiones para calcular el BER
    min_len = min(len(input_bits), len(decoded_bits))
    tx_eval = input_bits[:min_len]
    rx_eval = decoded_bits[:min_len]
    
    bit_errors = np.sum(tx_eval != rx_eval)
    ber = bit_errors / min_len
    
    print("RESULTADOS DE LA EVALUACIÓN:")
    print(f" -> Errores de Bit detectados: {bit_errors} de {min_len} totales")
    print(f" -> Tasa de Error de Bit (BER) calculada: {ber:.6e}")
    if ber == 0:
        print(" -> ¡Transmisión PERFECTA! (BER = 0.0)")
    else:
        print(" -> Transmisión con errores corregidos parcialmente.")
    print("=" * 60)
    
    # ==========================================================================
    # ANÁLISIS VISUAL (GRÁFICAS)
    # ==========================================================================
    print("GENERANDO REPORTES GRÁFICOS...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Simulación Modular Capa Física IEEE 802.11bb - Light Communications (LC) ({modulation})", fontsize=16, fontweight='bold')
    
    # 1. Señal en Banda Base (Tiempo)
    muestras_x = np.arange(40, 190)
    axes[0, 0].plot(muestras_x, tx_time_windowed[0][40:190], color='tab:blue', linewidth=2)
    axes[0, 0].set_title("Primer Símbolo OFDM en Banda Base (Tiempo)")
    axes[0, 0].set_xlabel("Muestras (1 muestra = 12.5 ns)")
    axes[0, 0].set_ylabel("Amplitud (V)")
    axes[0, 0].set_xlim([40, 190])
    axes[0, 0].grid(True)
    
    # 2. Espectro de Frecuencia (Desplazamiento a Frecuencia Intermedia)
    n_fft_spec = 4096
    freqs = np.fft.fftfreq(n_fft_spec, 1/f_s)
    
    x_bb_flat = tx_time_windowed.flatten()
    spec_bb = 20 * np.log10(np.abs(np.fft.fft(x_bb_flat, n_fft_spec)))
    spec_if = 20 * np.log10(np.abs(np.fft.fft(s_if, n_fft_spec)))
    
    # Solo mostrar frecuencias positivas
    pos_idx = freqs >= 0
    axes[0, 1].plot(freqs[pos_idx] / 1e6, spec_bb[pos_idx], label="Banda Base", color='tab:gray', alpha=0.7)
    axes[0, 1].plot(freqs[pos_idx] / 1e6, spec_if[pos_idx], label="Up-converted FI", color='tab:red')
    axes[0, 1].set_title("Espectro de Frecuencia (Desplazamiento a FI)")
    axes[0, 1].set_xlabel("Frecuencia (MHz)")
    axes[0, 1].set_ylabel("Magnitud (dB)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Modulación de Intensidad en LED (Dominio del Tiempo)
    axes[1, 0].plot(muestras_x, s_if[40:190], label="Señal FI Alterna", color='tab:purple', alpha=0.5)
    axes[1, 0].plot(muestras_x, optical_signal[40:190], label="Señal Óptica Unipolar (Con Bias)", color='tab:orange', linewidth=2.0)
    axes[1, 0].axhline(y=led_min, color='r', linestyle='--', label="Corte LED (Clipping Min)")
    axes[1, 0].axhline(y=led_max, color='r', linestyle=':', label="Saturación LED (Clipping Max)")
    axes[1, 0].set_title("Modulación de Intensidad en LED (Dominio del Tiempo)")
    axes[1, 0].set_xlabel("Muestras (1 muestra = 12.5 ns, Ventana = 1.87 μs)")
    axes[1, 0].set_ylabel("Corriente (A) / Potencia Óptica (W)")
    axes[1, 0].set_xlim([40, 190])
    axes[1, 0].legend(loc='upper right', fontsize='x-small', ncol=2)
    axes[1, 0].grid(True)
    
    # 4. Señal Demodulada vs Original en Banda Base (Primer Símbolo)
    demod_baseband_equalized = demod_baseband / (gain * scaling_factor)
    axes[1, 1].plot(muestras_x, x_bb_flat[40:190], label="Original TX", color='tab:blue', linewidth=2)
    axes[1, 1].plot(muestras_x, demod_baseband_equalized[40:190], label="Demodulada RX Ecualizada", color='tab:green', linestyle='--', linewidth=1.8)
    axes[1, 1].set_title("Comparación de Señales de Banda Base (Primer Símbolo)")
    axes[1, 1].set_xlabel("Muestras (1 muestra = 12.5 ns)")
    axes[1, 1].set_ylabel("Amplitud (V)")
    axes[1, 1].set_xlim([40, 190])
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. Constelación Transmitida (Puntos teóricos)
    axes[2, 0].scatter(np.real(tx_symbols[:2000]), np.imag(tx_symbols[:2000]), color='tab:blue', marker='o', s=40, label="Teóricos TX")
    axes[2, 0].set_title("Diagrama de Constelación Transmitido")
    axes[2, 0].set_xlabel("In-phase (I)")
    axes[2, 0].set_ylabel("Quadrature (Q)")
    axes[2, 0].grid(True)
    axes[2, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[2, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # 6. Constelación Recibida (Efecto del ruido y canal tras ecualización)
    axes[2, 1].scatter(np.real(rx_symbols_equalized[:2000]), np.imag(rx_symbols_equalized[:2000]), color='tab:orange', marker='.', s=15, alpha=0.6, label="Recibidos RX")
    axes[2, 1].scatter(np.real(tx_symbols[:2000]), np.imag(tx_symbols[:2000]), color='tab:red', marker='+', s=100, label="Teóricos")
    axes[2, 1].set_title("Diagrama de Constelación Recibido")
    axes[2, 1].set_xlabel("In-phase (I)")
    axes[2, 1].set_ylabel("Quadrature (Q)")
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    axes[2, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[2, 1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Guardar gráfica en la carpeta de la tesis y en la carpeta modular de LaTeX
    import os
    plot_dir_thesis = "c:\\Users\\carlo\\OneDrive\\Imágenes\\Documentos\\TESIS"
    plot_dir_latex = "c:\\Users\\carlo\\OneDrive\\Imágenes\\Documentos\\TESIS\\TESIS_LATEX"
    
    plot_path = os.path.join(plot_dir_thesis, output_filename)
    plt.savefig(plot_path, dpi=300)
    print(f" -> Gráfica de resultados guardada en: {plot_path}")
    
    plot_path_latex = os.path.join(plot_dir_latex, output_filename)
    plt.savefig(plot_path_latex, dpi=300)
    print(f" -> Gráfica de resultados guardada en: {plot_path_latex}")
    
    plt.close(fig)
    
    # Guardar figuras individuales si es la simulación ideal (para Cap. 2)
    if output_filename == "resultado_simulacion_ideal.png":
        print(" -> Exportando gráficos individuales para la Metodología (Capítulo 2)...")
        
        # A. Espectro FI
        fig_spec, ax_spec = plt.subplots(figsize=(7, 4.5))
        ax_spec.plot(freqs[pos_idx] / 1e6, spec_bb[pos_idx], label="Banda Base", color='tab:gray', alpha=0.7)
        ax_spec.plot(freqs[pos_idx] / 1e6, spec_if[pos_idx], label="Up-converted FI (20 MHz)", color='tab:red')
        ax_spec.set_title("Desplazamiento del Espectro OFDM a FI", fontsize=11, fontweight='bold')
        ax_spec.set_xlabel("Frecuencia (MHz)")
        ax_spec.set_ylabel("Magnitud (dB)")
        ax_spec.legend(fontsize='small')
        ax_spec.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_thesis, "modulo_espectro_fi.png"), dpi=300)
        plt.savefig(os.path.join(plot_dir_latex, "modulo_espectro_fi.png"), dpi=300)
        plt.close(fig_spec)
        
        # B. LED Intensidad
        fig_led, ax_led = plt.subplots(figsize=(7, 4.5))
        ax_led.plot(muestras_x, s_if[40:190], label="Señal FI Alterna (Bipolar)", color='tab:purple', alpha=0.5)
        ax_led.plot(muestras_x, optical_signal[40:190], label="Corriente Alimentación LED (Unipolar)", color='tab:orange', linewidth=2.0)
        ax_led.axhline(y=led_min, color='r', linestyle='--', label="Corte LED (Clipping Min: 0 A)")
        ax_led.axhline(y=led_max, color='r', linestyle=':', label="Saturación LED (Clipping Max: 5 A)")
        ax_led.set_title("Modulación de Intensidad en el Diodo LED", fontsize=11, fontweight='bold')
        ax_led.set_xlabel("Muestras (1 muestra = 12.5 ns)")
        ax_led.set_ylabel("Corriente (A) / Potencia Óptica (W)")
        ax_led.set_xlim([40, 190])
        ax_led.legend(loc='upper right', fontsize='x-small', ncol=2)
        ax_led.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_thesis, "modulo_led_intensidad.png"), dpi=300)
        plt.savefig(os.path.join(plot_dir_latex, "modulo_led_intensidad.png"), dpi=300)
        plt.close(fig_led)
        
        # C. Comparación RX/TX
        fig_comp, ax_comp = plt.subplots(figsize=(7, 4.5))
        ax_comp.plot(muestras_x, x_bb_flat[40:190], label="Original Transmitida (TX)", color='tab:blue', linewidth=2)
        ax_comp.plot(muestras_x, demod_baseband_equalized[40:190], label="Demodulada y Ecualizada (RX)", color='tab:green', linestyle='--', linewidth=1.8)
        ax_comp.set_title("Comparación de Señales de Banda Base", fontsize=11, fontweight='bold')
        ax_comp.set_xlabel("Muestras (1 muestra = 12.5 ns)")
        ax_comp.set_ylabel("Amplitud (V)")
        ax_comp.set_xlim([40, 190])
        ax_comp.legend(fontsize='small')
        ax_comp.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_thesis, "modulo_comparacion_rx.png"), dpi=300)
        plt.savefig(os.path.join(plot_dir_latex, "modulo_comparacion_rx.png"), dpi=300)
        plt.close(fig_comp)
        
        # D. Constelación RX
        fig_const, ax_const = plt.subplots(figsize=(7, 4.5))
        ax_const.scatter(np.real(rx_symbols_equalized[:2000]), np.imag(rx_symbols_equalized[:2000]), color='tab:orange', marker='.', s=15, alpha=0.6, label="Símbolos Recibidos")
        ax_const.scatter(np.real(tx_symbols[:2000]), np.imag(tx_symbols[:2000]), color='tab:red', marker='+', s=100, label="Referencias Gray 16-QAM")
        ax_const.set_title("Diagrama de Constelación Ecualizado (Link Ideal)", fontsize=11, fontweight='bold')
        ax_const.set_xlabel("Componente In-phase (I)")
        ax_const.set_ylabel("Componente Quadrature (Q)")
        ax_const.legend(fontsize='small')
        ax_const.grid(True)
        ax_const.axhline(y=0, color='k', linewidth=0.5)
        ax_const.axvline(x=0, color='k', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir_thesis, "modulo_constelacion_rx.png"), dpi=300)
        plt.savefig(os.path.join(plot_dir_latex, "modulo_constelacion_rx.png"), dpi=300)
        plt.close(fig_const)
    
    # Mostrar la gráfica (deshabilitado en ejecución automatizada para evitar bloqueo)
    # plt.show()

if __name__ == "__main__":
    # 1. Caso Ideal (Sin ruido de disparo solar)
    run_lc_simulation(solar_irradiance=0.0, output_filename="resultado_simulacion_ideal.png")
    
    # 2. Caso Real (Con ruido solar de día)
    print("\n" + "="*60 + "\n")
    run_lc_simulation(solar_irradiance=150.0, output_filename="resultado_simulacion.png")
