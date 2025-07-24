# Punto 2: Simulación de Difracción de Fraunhofer mediante la Transformada de Fourier

Este directorio contiene la solución para el segundo punto del parcial, el cual requiere el desarrollo de un programa que calcule y muestre el patrón de difracción 2D de cualquier abertura en la aproximación de Fraunhofer, utilizando la transformada de Fourier espacial.

Se incluyen dos scripts de Python que cumplen con este requisito:
1.  `Punto_2_solcuion.py`: Una implementación directa y clara del simulador.
2.  `Punto_2.py`: Una versión más avanzada con una interfaz gráfica enriquecida y más funcionalidades.

Ambos scripts están completamente documentados en español para facilitar su comprensión.

## Fundamento Teórico

### Difracción de Fraunhofer

La difracción es un fenómeno ondulatorio que ocurre cuando una onda se encuentra con un obstáculo o una abertura. La difracción de Fraunhofer, o de campo lejano, es un caso particular que se observa cuando la pantalla de observación está muy lejos de la abertura, o cuando se utiliza una lente para enfocar el patrón de difracción en su plano focal.

En estas condiciones, la distribución de amplitud del campo eléctrico (U) en el plano de observación (x', y') es proporcional a la **Transformada de Fourier 2D** de la función de transmitancia de la abertura (A), evaluada en las frecuencias espaciales correspondientes.

La relación matemática es la siguiente:

`U(x', y') ∝ ∬ A(x, y) * exp[-i * 2π * (fx*x + fy*y)] dx dy`

Donde:
- `A(x, y)` es la función de la abertura (1 si es transparente, 0 si es opaca).
- `(x, y)` son las coordenadas en el plano de la abertura.
- `(x', y')` son las coordenadas en el plano de observación.
- `fx = x' / (λ * z)` y `fy = y' / (λ * z)` son las frecuencias espaciales, con `λ` siendo la longitud de onda y `z` la distancia al plano de observación.

### Simulación Numérica con la Transformada Rápida de Fourier (FFT)

Analíticamente, resolver la integral de Fourier puede ser complejo o imposible para aberturas de formas arbitrarias. Sin embargo, numéricamente, podemos aprovechar que la Transformada de Fourier Discreta (DFT) es una excelente aproximación de la integral de Fourier.

El algoritmo de la **Transformada Rápida de Fourier (FFT)** es una implementación computacionalmente eficiente de la DFT. Este es el método utilizado en los scripts:

1.  **Representación de la Abertura:** La abertura se representa como una matriz 2D (una imagen digital), donde cada celda (píxel) tiene un valor (e.g., 1 para transparente, 0 para opaco).
2.  **Cálculo de la FFT:** Se aplica la función `fft2` de la librería `scipy.fft` a la matriz de la abertura. Esto calcula la amplitud y fase del campo difractado en el dominio de la frecuencia espacial.
3.  **Cálculo de la Intensidad:** La intensidad de la luz que se observa en la pantalla es proporcional al módulo al cuadrado de la amplitud del campo complejo: `I = |U|^2`.
4.  **Visualización:** El resultado, una matriz 2D de intensidades, se grafica como una imagen, representando el patrón de difracción.

Este método es extremadamente potente porque permite calcular el patrón de difracción para **cualquier forma de abertura** que pueda ser dibujada en una matriz, sin necesidad de resolver complejas integrales analíticas.

## Requisitos e Instalación

Para ejecutar los simuladores, es necesario tener Python instalado junto con las siguientes librerías:

- `numpy`
- `matplotlib`
- `scipy`

Puede instalar todas las dependencias fácilmente ejecutando el siguiente comando en su terminal:

```bash
pip install -r requirements.txt
```

## Instrucciones de Uso

### 1. Simulador Principal (`Punto_2_solcuion.py`)

Este script es la solución recomendada para la evaluación. Ofrece una interfaz clara con tres paneles: la abertura, el patrón de difracción 2D y un perfil de intensidad.

**Para ejecutarlo:**

```bash
python "Punto 2/Punto_2_solcuion.py"
```

**Características:**
- **Selección de Abertura:** Use los botones de radio a la izquierda para cambiar entre diferentes tipos de aberturas, incluyendo formas convencionales (círculo, rectángulo) y no convencionales (flor, gato).
- **Parámetros Globales:** Ajuste la longitud de onda, la distancia a la pantalla (z) y el nivel de zoom del patrón de difracción.
- **Parámetros Específicos:** Para cada tipo de abertura, aparecerán sliders específicos que permiten modificar sus dimensiones (e.g., diámetro, ancho, separación, etc.).
- **Visualización Triple:** Observe simultáneamente la forma de la abertura que está creando, el patrón de difracción 2D resultante y un corte transversal de la intensidad.

### 2. Simulador Avanzado (`Punto_2.py`)

Esta es una versión más elaborada con una interfaz gráfica que simula un montaje experimental completo.

**Para ejecutarlo:**

```bash
python "Punto 2/Punto_2.py"
```

**Características:**
- **Interfaz Esquemática:** Visualiza un láser, la abertura y la pantalla para una comprensión más intuitiva del experimento.
- **Controles Dinámicos:** La interfaz es altamente interactiva, con sliders y botones que actualizan la simulación en tiempo real.
- **Aberturas Complejas:** Incluye la capacidad de simular rejillas de difracción con desorden aleatorio, permitiendo estudiar efectos más avanzados.
- **Perfil de Intensidad Coloreado:** El perfil de intensidad se colorea según la longitud de onda seleccionada.
- **Compatibilidad con Notebooks:** Incluye código para una mejor integración en entornos como Jupyter o Google Colab.
