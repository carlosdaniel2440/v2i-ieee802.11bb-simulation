# Simulador de la Capa Física (LC PHY) IEEE 802.11bb para Enlaces V2I

Este directorio contiene la implementación limpia y modular en Python del simulador de la capa física de comunicaciones por luz visible (VLC) de alta eficiencia, desarrollado siguiendo las especificaciones de la enmienda **IEEE 802.11bb-2023** en enlaces de movilidad Vehículo a Infraestructura (V2I).

---

## 🛠️ Estructura del Código

El simulador está organizado en los siguientes archivos lógicos:

1. **`physical_layer.py`**:
   - Biblioteca de procesamiento digital de la capa física (transmisor y receptor).
   - Implementa: Codificador FEC Convolucional ($K=3$), Punzado (*Puncturing*) para tasas $2/3$, $3/4$ y $5/6$, Entrelazador de bloques, Asignación de Constelaciones Gray (de BPSK a 1024-QAM), Simetría Hermitiana, IFFT/FFT, Modulación/Demodulación I/Q en frecuencia intermedia (20 MHz), Filtro pasabajos Butterworth, Bias adaptativo, Recorte de LED (*Clipping*), Canal óptico geométrico/climático (Lambertiano + Beer-Lambert), y Ecualización en frecuencia de un toque (Single-Tap FEQ).

2. **`config_entorno.py`**:
   - Archivo de configuración global centralizada del simulador. Contiene los parámetros del LED, fotodiodo RSU, distancias de barrido, anchos de banda, varianza del ruido térmico y la especificación de irradiancia solar y atenuación para los cuatro escenarios climáticos (Noche Despejada, Día Soleado, Lluvia Moderada, Neblina Moderada).

3. **`test_phy.py`**:
   - Script principal de pruebas unitarias y caracterización de rendimiento.
   - Ejecuta validaciones funcionales de simetría hermitiana, pruebas ideales libres de ruido para todos los MCS y simulaciones de Monte Carlo para curvas de BER vs. SNR, BER vs. Distancia, y Throughput vs. Distancia en los cuatro escenarios ambientales.

4. **`test_phy_grid.py`**:
   - Módulo de pruebas avanzadas que realiza simulaciones paralelas y genera matrices comparativas de rendimiento del throughput y BER para diferentes anchos de Intervalo de Guarda (GI).

5. **`main.py`**:
   - Motor de integración para flujos de simulación por consola y generación de gráficas detalladas paso a paso en el dominio del tiempo y de la frecuencia.

---

## 🚀 Cómo Ejecutar el Simulador

### Requisitos Previos

Asegúrate de tener instalado Python 3.8+ y las siguientes bibliotecas científicas de Python:

```bash
pip install numpy scipy matplotlib
```

### Ejecutar Pruebas y Simulaciones

Para ejecutar las validaciones lógicas y generar las curvas estadísticas de BER y Throughput en el canal exterior:

```bash
python test_phy.py
```

Para ejecutar las simulaciones comparativas de Intervalos de Guarda bajo movilidad:

```bash
python test_phy_grid.py
```
