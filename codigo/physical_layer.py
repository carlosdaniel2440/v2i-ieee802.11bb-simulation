"""
physical_layer.py
Simulación modular de la capa física (PHY) bajo el estándar IEEE 802.11bb (Light Communications).
Implementa OFDM con simetría hermitiana (DCO-OFDM), modulación a Frecuencia Intermedia (FI),
canal óptico Lambertiano, y decodificación con algoritmo de Viterbi.
"""

import numpy as np
import scipy.signal

# ==============================================================================
# 1. FUNCIONES DEL TRANSMISOR (TX)
# ==============================================================================

def fec_encoder(bits: np.ndarray, code_rate: str = "1/2") -> np.ndarray:
    """
    Codificador FEC Convolucional (Tasa 1/2, K=3).
    Polinomios generadores: G0 = 7 (111 en binario), G1 = 5 (101 en binario).
    """
    state = [0, 0] # Estado inicial [u[n-1], u[n-2]]
    encoded = []
    
    for bit in bits:
        # Salidas XOR según los polinomios generadores
        v0 = bit ^ state[0] ^ state[1]
        v1 = bit ^ state[1]
        encoded.extend([v0, v1])
        # Actualización de estado: u[n-1] = bit, u[n-2] = anterior u[n-1]
        state = [bit, state[0]]
        
    return np.array(encoded, dtype=int)


def puncture(bits: np.ndarray, rate: str) -> np.ndarray:
    """
    Aplica punzado (puncturing) a los bits codificados de tasa 1/2 para obtener
    tasas de código mayores (2/3, 3/4, 5/6) según el estándar.
    """
    if rate == "1/2":
        return bits
    elif rate == "2/3":
        mask = [1, 1, 1, 0]
    elif rate == "3/4":
        mask = [1, 1, 1, 0, 0, 1]
    elif rate == "5/6":
        mask = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    else:
        raise ValueError(f"Tasa de código FEC '{rate}' no soportada.")
    
    n_bits = len(bits)
    mask_len = len(mask)
    full_mask = np.tile(mask, int(np.ceil(n_bits / mask_len)))[:n_bits]
    return bits[full_mask == 1]


def depuncture(bits: np.ndarray, rate: str, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruye la secuencia de bits de longitud target_len insertando ceros (erasure)
    en las posiciones que fueron eliminadas por el punzado en el transmisor.
    Retorna una tupla (depunctured_bits, is_erasure).
    """
    if rate == "1/2":
        return bits, np.zeros(len(bits), dtype=bool)
    elif rate == "2/3":
        mask = [1, 1, 1, 0]
    elif rate == "3/4":
        mask = [1, 1, 1, 0, 0, 1]
    elif rate == "5/6":
        mask = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    else:
        raise ValueError(f"Tasa de código FEC '{rate}' no soportada.")
    
    mask_len = len(mask)
    full_mask = np.tile(mask, int(np.ceil(target_len / mask_len)))[:target_len]
    
    depunctured = np.zeros(target_len, dtype=bits.dtype)
    is_erasure = np.zeros(target_len, dtype=bool)
    
    # Colocar los bits recibidos en las posiciones donde la máscara es 1
    depunctured[full_mask == 1] = bits
    # Las posiciones donde la máscara es 0 se marcan como borrados (erasures)
    is_erasure[full_mask == 0] = True
    
    return depunctured, is_erasure



def interleave(bits: np.ndarray, n_cols: int = 16) -> np.ndarray:
    """
    Interleaver de bloque: Escribe fila por fila, lee columna por columna.
    Ayuda a dispersar ráfagas de errores en el canal.
    """
    n_bits = len(bits)
    # Rellenar con ceros si no es múltiplo de n_cols
    remainder = n_bits % n_cols
    pad_len = 0
    if remainder != 0:
        pad_len = n_cols - remainder
        bits = np.concatenate((bits, np.zeros(pad_len, dtype=int)))
    
    n_rows = len(bits) // n_cols
    matrix = bits.reshape((n_rows, n_cols))
    # Transponer para leer columna por columna
    interleaved = matrix.T.flatten()
    
    # Guardar metadato del tamaño original si es necesario, o simplemente retornar
    return interleaved


def constellation_mapping(bits: np.ndarray, scheme: str = "QPSK") -> np.ndarray:
    """
    Mapeo de bits a símbolos complejos usando codificación Gray.
    Soporta: BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM.
    """
    scheme = scheme.upper()
    
    if scheme == "BPSK":
        mapping = {0: -1.0, 1: 1.0}
        return np.array([mapping[b] for b in bits], dtype=complex)
        
    qam_sizes = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
    if scheme not in qam_sizes:
        raise ValueError(f"Esquema de modulación '{scheme}' no soportado.")
        
    m_bits = qam_sizes[scheme]
    remainder = len(bits) % m_bits
    if remainder != 0:
        bits = np.pad(bits, (0, m_bits - remainder), 'constant')
        
    k = m_bits // 2
    reshaped = bits.reshape(-1, m_bits)
    symbols = []
    m_val = 1 << k
    
    for sym_bits in reshaped:
        i_bits = sym_bits[:k]
        q_bits = sym_bits[k:]
        
        g_i = 0
        for b in i_bits:
            g_i = (g_i << 1) | b
        g_q = 0
        for b in q_bits:
            g_q = (g_q << 1) | b
            
        b_i = 0
        temp = g_i
        while temp > 0:
            b_i ^= temp
            temp >>= 1
            
        b_q = 0
        temp = g_q
        while temp > 0:
            b_q ^= temp
            temp >>= 1
            
        i_val = 2 * b_i - m_val + 1
        q_val = 2 * b_q - m_val + 1
        symbols.append(i_val + 1j * q_val)
        
    norm_factor = np.sqrt((2.0 / 3.0) * ((1 << (2 * k)) - 1))
    return np.array(symbols) / norm_factor



def apply_hermitian_symmetry(complex_symbols: np.ndarray, fft_size: int = 256, active_carriers: list = None) -> np.ndarray:
    """
    Aplica Simetría Hermitiana a los símbolos complejos para OFDM en Light Communications (LC) (DCO-OFDM).
    Subportadoras activas de datos: por defecto 6 a 30 (ajustadas para 80 MHz fs, 20 MHz f_if).
    Subportadoras de guarda: 0 (DC), 1-5 (Guardas DC), Nyquist, y equivalentes conjugados.
    Retorna una matriz de forma (num_ofdm_symbols, fft_size) que entrará a la IFFT.
    """
    n_half = fft_size // 2
    if active_carriers is None:
        active_carriers = list(range(6, 31)) # Por defecto 6 a 30
    n_data_per_symbol = len(active_carriers)
    
    # Rellenar con ceros si no hay suficientes símbolos de datos para completar el último bloque
    remainder = len(complex_symbols) % n_data_per_symbol
    if remainder != 0:
        pad_len = n_data_per_symbol - remainder
        complex_symbols = np.concatenate((complex_symbols, np.zeros(pad_len, dtype=complex)))
        
    num_symbols = len(complex_symbols) // n_data_per_symbol
    ofdm_freq_matrix = np.zeros((num_symbols, fft_size), dtype=complex)
    
    for i in range(num_symbols):
        # Tomar trozo de datos para este símbolo
        symbol_data = complex_symbols[i * n_data_per_symbol : (i + 1) * n_data_per_symbol]
        
        # Mapear datos a subportadoras en la mitad positiva del espectro
        ofdm_freq_matrix[i, active_carriers] = symbol_data
        
        # Aplicar Simetría Hermitiana para la mitad negativa
        # X[N - k] = conj(X[k]) para k = 1 a N/2 - 1
        for k in range(1, n_half):
            ofdm_freq_matrix[i, fft_size - k] = np.conj(ofdm_freq_matrix[i, k])
            
        # Las portadoras 0 y N/2 quedan en 0 por inicialización
        
    return ofdm_freq_matrix


def idft_processing(freq_data: np.ndarray) -> np.ndarray:
    """
    Transformada Rápida de Fourier Inversa (IFFT) aplicada a lo largo de las filas.
    Verifica que la salida sea estrictamente real y extrae la parte real.
    """
    # IFFT por filas
    time_signal = np.fft.ifft(freq_data, axis=1)
    
    # Verificación de componente imaginaria residual
    max_imag = np.max(np.abs(np.imag(time_signal)))
    if max_imag > 1e-10:
        print(f"Advertencia: Componente imaginaria residual detectada en IFFT ({max_imag:.2e}).")
        
    return np.real(time_signal)


def insert_guard_interval(time_signal: np.ndarray, cp_len: int = 64) -> np.ndarray:
    """
    Inserta un Intervalo de Guarda usando un Prefijo Cíclico (CP).
    Copia las últimas 'cp_len' muestras de cada símbolo y las coloca al inicio.
    """
    # time_signal es de tamaño (num_symbols, fft_size)
    cp_part = time_signal[:, -cp_len:]
    time_signal_cp = np.hstack((cp_part, time_signal))
    return time_signal_cp


def windowing(time_signal_cp: np.ndarray, roll_off: float = 0.05) -> np.ndarray:
    """
    Aplica una ventana de coseno alzado (Raised Cosine) a cada símbolo OFDM
    para suavizar las transiciones temporales y reducir las emisiones fuera de banda.
    """
    num_symbols, symbol_len = time_signal_cp.shape
    w = np.ones(symbol_len)
    
    # Longitud de la zona de transición
    transition_len = int(np.round(roll_off * symbol_len))
    if transition_len > 0:
        # Flanco de subida
        w[:transition_len] = 0.5 * (1 - np.cos(np.pi * np.arange(transition_len) / transition_len))
        # Flanco de bajada
        w[-transition_len:] = 0.5 * (1 - np.cos(np.pi * np.arange(transition_len-1, -1, -1) / transition_len))
        
    # Aplicar ventana por filas
    windowed_signal = time_signal_cp * w
    return windowed_signal


def iq_modulation(baseband_matrix: np.ndarray, f_if: float, f_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Modulación I/Q (Up-conversion) de la señal real de banda base a Frecuencia Intermedia (FI).
    Concatenamos los símbolos OFDM en un stream 1D y multiplicamos por la portadora cos(2*pi*f_if*t).
    Retorna la señal de Frecuencia Intermedia y el vector de tiempo correspondiente.
    """
    # Concatenar todos los símbolos OFDM en un vector 1D
    x_baseband = baseband_matrix.flatten()
    
    # Crear vector de tiempo
    t = np.arange(len(x_baseband)) / f_s
    
    # Up-conversion
    s_if = x_baseband * np.cos(2 * np.pi * f_if * t)
    
    return s_if, t


def add_dc_bias(s_if: np.ndarray, bias_mode: str = "adaptive", fixed_bias: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Añade una corriente de polarización constante (DC Bias) a la señal de FI
    para hacerla estrictamente no negativa (unipolar), requisito de los emisores LED.
    """
    if bias_mode == "fixed":
        bias = fixed_bias
    elif bias_mode == "adaptive":
        # Se calcula para que el mínimo de la señal quede ligeramente sobre cero
        bias = -np.min(s_if) + 0.05
    else:
        bias = 0.0
        
    s_biased = s_if + bias
    return s_biased, bias


def tx_optical_front_end(s_biased: np.ndarray, led_min: float = 0.0, led_max: float = 2.0) -> np.ndarray:
    """
    Frente Óptico del Transmisor (LED Driver + LED).
    Convierte la corriente eléctrica polarizada en intensidad luminosa,
    aplicando recorte (clipping) si se exceden los límites lineales del LED.
    """
    # El LED corta por debajo del umbral de encendido (led_min) y se satura en led_max
    optical_power = np.clip(s_biased, led_min, led_max)
    return optical_power


# ==============================================================================
# 2. CANAL ÓPTICO
# ==============================================================================
def optical_wireless_channel(optical_signal: np.ndarray, distance: float = 10.0, 
                             rx_area: float = 1e-4, semi_angle_led: float = 60.0, 
                             fov_rx: float = 60.0, index_conc: float = 1.5,
                             tx_filter_gain: float = 1.0, noise_variance: float = 2.78e-16,
                             solar_irradiance: float = 0.0, bandwidth: float = 40e6,
                             extinction_coef: float = 0.0, phi: float = 0.0,
                             psi: float = 0.0) -> np.ndarray:
    """
    Modelo de propagación óptico de línea de vista (LOS) Lambertiano.
    Permite modelar canal ideal (solar_irradiance = 0.0) usando una varianza de ruido fija,
    o canal real con ruido de disparo solar (solar_irradiance > 0.0) y ruido térmico.
    También incorpora atenuación atmosférica (lluvia, neblina) mediante extinction_coef.
    """
    # 1. Calcular el orden de Lambert m
    phi_half = np.radians(semi_angle_led)
    m = -np.log(2.0) / np.log(np.cos(phi_half))
    
    # Ángulos de irradiancia e incidencia en radianes
    phi_rad = np.radians(phi)
    psi_rad = np.radians(psi)
    
    # Campo de visión (FOV) del receptor
    fov_rad = np.radians(fov_rx)
    
    if psi_rad > fov_rad:
        # Fuera del campo de visión del receptor
        H_0 = 0.0
    else:
        # Ganancia del concentrador óptico g(psi)
        g_psi = (index_conc**2) / (np.sin(fov_rad)**2)
        
        # Ganancia DC del canal H(0)
        H_0 = ((m + 1) * rx_area / (2 * np.pi * (distance**2))) * \
              (np.cos(phi_rad)**m) * tx_filter_gain * g_psi * np.cos(psi_rad)
              
    # 2. Atenuación óptica de la señal de datos (Lambertiana + Atmosférica)
    atmospheric_attenuation = np.exp(-extinction_coef * distance)
    received_optical = H_0 * atmospheric_attenuation * optical_signal

    
    # 3. Modelado y adición de ruido
    if solar_irradiance > 0.0:
        # a. Ganancia del concentrador óptico
        g_psi = (index_conc**2) / (np.sin(fov_rad)**2) if psi_rad <= fov_rad else 0.0
        
        # b. Potencia solar recibida en el fotodetector (después de concentrador y filtro)
        P_sol = solar_irradiance * rx_area * tx_filter_gain * g_psi
        
        # c. Corriente de fondo (DC) generada en el fotodiodo (responsividad = 0.6 A/W)
        R = 0.6
        I_bg = R * P_sol
        
        # d. Varianza de ruido de disparo eléctrico (Shot Noise)
        # q = 1.602e-19 C (carga del electrón)
        q = 1.602e-19
        sigma2_shot = 2 * q * I_bg * bandwidth
        
        # e. Varianza de ruido térmico (Thermal Noise)
        # 1e-16 A^2 es un valor de ruido térmico típico de un TIA a temperatura ambiente para este BW
        sigma2_thermal = 1e-16
        
        # f. Varianza total del ruido eléctrico
        sigma2_noise_electrical = sigma2_shot + sigma2_thermal
        
        # g. Varianza equivalente del ruido en el dominio óptico (antes del escalado de responsividad en RX)
        total_noise_variance = sigma2_noise_electrical / (R**2)
    else:
        # Si solar_irradiance == 0, usamos la varianza de ruido fija (canal ideal)
        total_noise_variance = noise_variance
        
    # Añadir ruido
    noise = np.random.normal(0, np.sqrt(total_noise_variance), len(received_optical))
    received_noisy = received_optical + noise
    
    return received_noisy


# ==============================================================================
# 3. FUNCIONES DEL RECEPTOR (RX)
# ==============================================================================

def rx_optical_front_end(received_signal: np.ndarray, responsivity: float = 0.6) -> np.ndarray:
    """
    Frente Óptico del Receptor (Fotodiodo + TIA).
    Convierte la potencia óptica incidente en corriente eléctrica proporcional.
    """
    electrical_current = responsivity * received_signal
    return electrical_current


def remove_dc_bias(electrical_current: np.ndarray) -> np.ndarray:
    """
    Bloqueo DC (Filtro paso altos).
    Remueve la corriente continua promedio para recuperar la señal de datos alterna.
    """
    # En la práctica se implementa con un capacitor en serie
    return electrical_current - np.mean(electrical_current)


def iq_demodulation(i_ac: np.ndarray, f_if: float, f_s: float, f_cutoff: float = 15.0e6) -> np.ndarray:
    """
    Demodulación I/Q (Down-conversion) de Frecuencia Intermedia a Banda Base.
    Multiplica por cos(2*pi*f_if*t) y aplica un filtro digital paso bajos Butterworth.
    """
    t = np.arange(len(i_ac)) / f_s
    
    # Down-conversion analógica (multiplicar por 2 para compensar la atenuación del carrier)
    mix = i_ac * np.cos(2 * np.pi * f_if * t) * 2.0
    
    # Filtro paso bajos Butterworth digital (Frecuencia de corte = f_cutoff)
    nyquist = 0.5 * f_s
    w_norm = f_cutoff / nyquist
    
    b, a = scipy.signal.butter(N=5, Wn=w_norm, btype='low')
    
    # Filtro de fase lineal de ida y vuelta
    demod_baseband = scipy.signal.filtfilt(b, a, mix)
    
    return demod_baseband


def remove_guard_interval(demod_baseband: np.ndarray, symbol_len: int, cp_len: int) -> np.ndarray:
    """
    Reconstruye la matriz de símbolos OFDM y descarta el Prefijo Cíclico.
    """
    total_samples = len(demod_baseband)
    single_symbol_cp_len = symbol_len + cp_len
    num_symbols = total_samples // single_symbol_cp_len
    
    # Truncar señal residual al final si la hay
    demod_baseband = demod_baseband[:num_symbols * single_symbol_cp_len]
    
    # Reestructurar a matriz (num_symbols, symbol_len + cp_len)
    matrix_cp = demod_baseband.reshape((num_symbols, single_symbol_cp_len))
    
    # Remover el CP del inicio de cada símbolo
    matrix_no_cp = matrix_cp[:, cp_len:]
    return matrix_no_cp


def dft_processing(time_data: np.ndarray) -> np.ndarray:
    """
    Transformada Rápida de Fourier (FFT) aplicada a lo largo de las filas.
    Transfiere los datos de tiempo de regreso al dominio de la frecuencia.
    """
    freq_data = np.fft.fft(time_data, axis=1)
    return freq_data


def remove_hermitian_symmetry(freq_data: np.ndarray, fft_size: int = 256, active_carriers: list = None) -> np.ndarray:
    """
    Extrae los símbolos de datos complejos de las subportadoras activas.
    Por defecto de la 6 a la 30.
    """
    if active_carriers is None:
        active_carriers = list(range(6, 31))
    
    # Extraer los datos por filas y aplanarlos en un único array 1D
    symbols_matrix = freq_data[:, active_carriers]
    complex_symbols = symbols_matrix.flatten()
    
    return complex_symbols


def constellation_demapping(complex_symbols: np.ndarray, scheme: str = "QPSK") -> np.ndarray:
    """
    Decodificador de constelación (demapper de decisión dura).
    Soporta: BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM.
    Optimizado en rendimiento mediante vectorización con NumPy.
    """
    scheme = scheme.upper()
    
    if scheme == "BPSK":
        return (np.real(complex_symbols) > 0).astype(int)
        
    qam_sizes = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}
    if scheme not in qam_sizes:
        raise ValueError(f"Esquema de modulación '{scheme}' no soportado.")
        
    m_bits = qam_sizes[scheme]
    k = m_bits // 2
    m_val = 1 << k
    norm_factor = np.sqrt((2.0 / 3.0) * ((1 << (2 * k)) - 1))
    
    s_denorm = complex_symbols * norm_factor
    i_val = np.real(s_denorm)
    q_val = np.imag(s_denorm)
    
    idx_i = np.round((i_val + m_val - 1) / 2.0).astype(int)
    idx_i = np.clip(idx_i, 0, m_val - 1)
    
    idx_q = np.round((q_val + m_val - 1) / 2.0).astype(int)
    idx_q = np.clip(idx_q, 0, m_val - 1)
    
    g_i = idx_i ^ (idx_i >> 1)
    g_q = idx_q ^ (idx_q >> 1)
    
    shifts = np.arange(k - 1, -1, -1)
    bits_i = (g_i[:, None] >> shifts) & 1
    bits_q = (g_q[:, None] >> shifts) & 1
    
    bits_symbol = np.hstack((bits_i, bits_q))
    return bits_symbol.flatten()



def deinterleave(bits: np.ndarray, n_cols: int = 16) -> np.ndarray:
    """
    Deinterleaver de bloque: Invierte la transposición realizada en interleave.
    Reorganiza los bits de vuelta a su orden secuencial original.
    """
    n_bits = len(bits)
    n_rows = n_bits // n_cols
    
    # La señal interleaved fue leída columna por columna
    # Reestructuramos la señal recibida en la forma transpuesta (n_cols, n_rows)
    matrix_t = bits.reshape((n_cols, n_rows))
    # Volvemos a transponer para obtener la matriz original (n_rows, n_cols)
    original_matrix = matrix_t.T
    
    return original_matrix.flatten()


def fec_decoder(coded_bits: np.ndarray, code_rate: str = "1/2", is_erasure: np.ndarray = None) -> np.ndarray:
    """
    Decodificador FEC Convolucional usando el Algoritmo de Viterbi (Decisión Dura).
    G0 = 7 (octal), G1 = 5 (octal).
    Resuelve el camino óptimo a través de la rejilla (trellis) con soporte para borrados (erasures).
    Optimizado a O(N) de tiempo y O(N) de memoria mediante almacenamiento de traceback.
    """
    num_states = 4
    n_steps = len(coded_bits) // 2
    
    state_transitions = {
        0: {0: (0, (0, 0)), 1: (2, (1, 1))}, # Estado 00
        1: {0: (0, (1, 1)), 1: (2, (0, 0))}, # Estado 01
        2: {0: (1, (1, 0)), 1: (3, (0, 1))}, # Estado 10
        3: {0: (1, (0, 1)), 1: (3, (1, 0))}  # Estado 11
    }
    
    path_metrics = np.full(num_states, np.inf)
    path_metrics[0] = 0.0
    
    # traceback[step][state] = (estado_predecesor, bit_de_entrada)
    traceback_state = np.zeros((n_steps, num_states), dtype=int)
    traceback_input = np.zeros((n_steps, num_states), dtype=int)
    
    for i in range(n_steps):
        r0 = coded_bits[2 * i]
        r1 = coded_bits[2 * i + 1]
        
        if is_erasure is not None:
            erasure0 = is_erasure[2 * i]
            erasure1 = is_erasure[2 * i + 1]
        else:
            erasure0 = False
            erasure1 = False
            
        new_metrics = np.full(num_states, np.inf)
        new_predecessors = np.zeros(num_states, dtype=int)
        new_inputs = np.zeros(num_states, dtype=int)
        
        for curr_state in range(num_states):
            if path_metrics[curr_state] == np.inf:
                continue
                
            for input_bit in [0, 1]:
                next_state, (v0, v1) = state_transitions[curr_state][input_bit]
                
                dist0 = 0 if erasure0 else (r0 ^ v0)
                dist1 = 0 if erasure1 else (r1 ^ v1)
                dist = dist0 + dist1
                metric = path_metrics[curr_state] + dist
                
                if metric < new_metrics[next_state]:
                    new_metrics[next_state] = metric
                    new_predecessors[next_state] = curr_state
                    new_inputs[next_state] = input_bit
                    
        path_metrics = new_metrics
        traceback_state[i, :] = new_predecessors
        traceback_input[i, :] = new_inputs
        
    best_state = np.argmin(path_metrics)
    
    # Reconstrucción del camino mediante traceback (de atrás hacia adelante)
    decoded_bits = np.zeros(n_steps, dtype=int)
    curr_state = best_state
    for i in range(n_steps - 1, -1, -1):
        decoded_bits[i] = traceback_input[i, curr_state]
        curr_state = traceback_state[i, curr_state]
        
    return decoded_bits


def get_mcs_params(mcs: int) -> tuple[str, str]:
    """
    Retorna (modulación, tasa de código) para un índice MCS (0-11) en IEEE 802.11bb.
    """
    mcs_table = {
        0: ("BPSK", "1/2"),
        1: ("QPSK", "1/2"),
        2: ("QPSK", "3/4"),
        3: ("16QAM", "1/2"),
        4: ("16QAM", "3/4"),
        5: ("64QAM", "2/3"),
        6: ("64QAM", "3/4"),
        7: ("64QAM", "5/6"),
        8: ("256QAM", "3/4"),
        9: ("256QAM", "5/6"),
        10: ("1024QAM", "3/4"),
        11: ("1024QAM", "5/6")
    }
    if mcs not in mcs_table:
        raise ValueError(f"Índice MCS {mcs} no soportado (debe ser de 0 a 11).")
    return mcs_table[mcs]
