# config_entorno.py
# FUENTE ÚNICA DE VERDAD: Parámetros del transceptor y canal vehicular V2I

import numpy as np

# ==============================================================================
# 1. CONFIGURACIÓN DEL TRANSCEPTOR (Estándar IEEE 802.11bb)
# ==============================================================================
FFT_SIZE = 1024
CP_LEN = 64
F_IF = 20.0e6       # Frecuencia Intermedia (20 MHz)
F_S = 80.0e6        # Frecuencia de muestreo (80 MHz)

# 234 portadoras de datos activas centradas en FI = 20 MHz
ACTIVE_CARRIERS = list(range(139, 256)) + list(range(257, 374))

# ==============================================================================
# 2. GEOMETRÍA Y PARÁMETROS DEL ENLACE VEHICULAR V2I (Semáforo)
# ==============================================================================
H_DIFF = 4.0         # Diferencia de altura (Semáforo a 5.5m - Receptor a 1.5m)
SEMI_ANGLE = 15.0    # Haz del faro vehicular directivo (semiángulo de 15 grados)
FOV_RX = 60.0        # Campo de visión (FOV) del receptor fotodetector
INDEX_CONC = 1.5     # Índice de refracción del lente concentrador
RX_AREA = 10e-4      # Área de captación del detector (10 cm^2)
RESPONSIVITY = 0.6   # Responsividad del fotodetector (A/W)
BANDWIDTH = 40.0e6   # Ancho de banda de ruido eléctrico (40 MHz)

# Límites de potencia óptica del LED vehicular
LED_MIN = 0.0
LED_MAX = 5.0
