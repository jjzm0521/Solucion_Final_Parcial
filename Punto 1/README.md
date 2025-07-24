# Punto 1: Difracción de Fraunhofer de una Abertura Compuesta

Este proyecto simula el patrón de difracción de Fraunhofer producido por una configuración de dos aberturas: una rendija rectangular y una abertura anular, separadas verticalmente por una distancia `D`.

## 1. Fundamento Teórico

La difracción de Fraunhofer, o de campo lejano, describe el comportamiento de las ondas cuando pasan a través de una abertura y se observan a una distancia suficientemente grande. Bajo esta aproximación, el patrón de difracción es la [Transformada de Fourier](https.es.wikipedia.org/wiki/Transformada_de_Fourier) de la función de la abertura.

### a) La Función de Abertura

Nuestra abertura está compuesta por dos formas:
1.  Un **rectángulo** de ancho `a` y alto `b`, centrado en `(0, D/2)`.
2.  Un **anillo** con radio interno `R1` y radio externo `R2`, centrado en `(0, -D/2)`.

La función que describe esta abertura, `T(x̃, ỹ)`, es 1 dentro de estas formas y 0 fuera.

### b) El Campo Eléctrico en el Plano de Observación

El campo eléctrico `E(x', y')` en el plano de observación es proporcional a la Transformada de Fourier de la función de la abertura `T(x̃, ỹ)`. Por el [principio de superposición](https.es.wikipedia.org/wiki/Principio_de_superposici%C3%B3n), podemos calcular el campo de cada abertura por separado y luego sumarlos, teniendo en cuenta sus respectivas posiciones.

-   **Campo del Rectángulo (`E_rect`)**: La transformada de Fourier de una función rectangular es una función **sinc** (seno cardinal).

    `E_rect ∝ (a * b) * sinc(k * a * x' / (2 * z)) * sinc(k * b * y' / (2 * z)) * exp(i * k * D * y' / (2 * z))`

    Donde `k = 2π/λ` es el número de onda y el término exponencial representa el desfase debido a la posición `D/2`.

-   **Campo del Anillo (`E_anillo`)**: Un anillo puede considerarse como la resta de los campos de dos círculos. La transformada de Fourier de un círculo es una [función de Bessel de primer orden](https.es.wikipedia.org/wiki/Funci%C3%B3n_de_Bessel) (`J1`).

    `E_anillo ∝ [π*R2^2 * (2*J1(k*R2*r'/z)/(k*R2*r'/z)) - π*R1^2 * (2*J1(k*R1*r'/z)/(k*R1*r'/z))] * exp(-i * k * D * y' / (2 * z))`

    Donde `r' = sqrt(x'^2 + y'^2)` y el término exponencial se debe a la posición `-D/2`.

### c) La Intensidad del Patrón de Difracción

La intensidad `I(x', y')` es proporcional al módulo al cuadrado del campo eléctrico total:

`I = |E_total|^2 = |E_rect + E_anillo|^2`

Al expandir esta expresión, obtenemos tres términos:

`I = |E_rect|^2 + |E_anillo|^2 + 2 * Re(E_rect * E_anillo^*)`

1.  `|E_rect|^2`: El patrón de difracción del rectángulo solo.
2.  `|E_anillo|^2`: El patrón de difracción del anillo solo.
3.  `2 * Re(...)`: El **término de interferencia**, que modula la suma de las intensidades individuales y depende de la separación `D` entre las aberturas. Es este término el que crea las franjas de interferencia características.

El simulador implementa esta ecuación completa para calcular la intensidad en cada punto del plano de observación.

## 2. Instrucciones de Uso

### a) Requisitos

Para ejecutar la simulación, necesita tener Python y las siguientes bibliotecas instaladas:

-   `numpy`
-   `matplotlib`
-   `scipy`

Puede instalar todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

### b) Ejecución

Para iniciar el simulador, ejecute el siguiente comando en su terminal:

```bash
python "Punto _1.py"
```

### c) Interfaz Gráfica

La interfaz del simulador se compone de:

-   **Panel de Control (Izquierda)**:
    -   **Visualización de la Abertura**: Muestra la forma y posición de las aberturas.
    -   **Sliders (Controles deslizantes)**: Permiten modificar en tiempo real los parámetros físicos del sistema:
        -   `λ (nm)`: Longitud de onda de la luz.
        -   `D (mm)`: Distancia entre los centros de las aberturas.
        -   `a (mm), b (mm)`: Dimensiones del rectángulo.
        -   `R1 (mm), R2 (mm)`: Radios del anillo.
        -   `z (m)`: Distancia al plano de observación.
        -   `Rango (mm)`: Zoom de la visualización del patrón de difracción.

-   **Panel de Visualización (Derecha)**:
    -   Muestra el **patrón de difracción 2D** calculado. La intensidad se representa en una escala de colores (del negro al color de la luz simulada) y se aplica una escala logarítmica para resaltar los detalles más tenues.
    -   Superpuestos en el gráfico se encuentran los valores actuales de los parámetros para una referencia rápida.

## 3. Análisis de Resultados

Al utilizar el simulador, se pueden observar varios fenómenos clave:

-   **Efecto de la longitud de onda (`λ`)**: A mayor longitud de onda, el patrón de difracción se expande.
-   **Efecto de la separación (`D`)**: Al aumentar `D`, las franjas de interferencia (el término `cos(k*D*y'/z)`) se vuelven más finas y juntas.
-   **Efecto del tamaño de las aberturas (`a, b, R1, R2`)**: Aberturas más pequeñas producen patrones de difracción más grandes (principio de incertidumbre).
-   **Condición de Campo Lejano (`z`)**: La aproximación de Fraunhofer es válida cuando `z >> (tamaño de la abertura)² / λ`. El simulador asume que esta condición se cumple. Al variar `z`, se puede ver cómo el tamaño del patrón en la pantalla cambia linealmente.
