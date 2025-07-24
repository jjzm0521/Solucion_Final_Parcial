# Punto 3: Difracción de Fresnel y la Transición al Campo Lejano

Este proyecto simula el patrón de difracción en el **régimen de Fresnel** (campo cercano) para aberturas unidimensionales. A diferencia de la difracción de Fraunhofer, la aproximación de Fresnel es válida a distancias más cortas de la abertura y revela una estructura de patrón más compleja. El simulador también permite visualizar la transición del régimen de Fresnel al de Fraunhofer al variar los parámetros.

## 1. Fundamento Teórico

### a) Difracción de Fresnel vs. Fraunhofer

La principal diferencia entre las dos aproximaciones radica en cómo se trata el término de fase en la integral de difracción de Huygens-Fresnel.

-   **Fraunhofer (Campo Lejano)**: Se desprecian los términos cuadráticos de la fase, asumiendo que el plano de observación está en el "infinito". Esto simplifica la integral a una Transformada de Fourier.
-   **Fresnel (Campo Cercano)**: Se conservan los términos cuadráticos de la fase. Esto resulta en una integral más compleja que no es una simple Transformada de Fourier.

La validez de una u otra aproximación se cuantifica mediante el **Número de Fresnel (F)**:

`F = a² / (λz)`

Donde `a` es el tamaño característico de la abertura, `λ` es la longitud de onda y `z` es la distancia de observación.

-   `F >> 1`: Régimen de óptica geométrica (la sombra proyectada es nítida).
-   `F ≈ 1`: **Régimen de Fresnel**. Los patrones son complejos y dependen fuertemente de la distancia `z`.
-   `F << 1`: **Régimen de Fraunhofer**. El patrón de difracción ya no cambia de forma, solo escala linealmente con la distancia `z`.

### b) Las Integrales de Fresnel y la Espiral de Cornu

La solución a la integral de difracción de Fresnel para aberturas rectangulares se puede expresar en términos de dos funciones especiales: las **integrales de Fresnel**, `C(u)` y `S(u)`.

`C(u) = ∫ cos(π/2 * t²) dt` (desde 0 hasta u)
`S(u) = ∫ sin(π/2 * t²) dt` (desde 0 hasta u)

El campo eléctrico complejo `E(x')` en un punto de la pantalla se calcula combinando estas funciones evaluadas en los límites de la abertura. Por ejemplo, para una rendija simple, el campo es `E ∝ [C(u2)-C(u1)] + i*[S(u2)-S(u1)]`.

Estas integrales tienen una representación gráfica muy elegante conocida como la **Espiral de Cornu** (o Clotoide). La diferencia vectorial entre dos puntos de la espiral da directamente la amplitud y fase del campo eléctrico. Aunque este simulador usa la implementación numérica de `scipy.special.fresnel`, la espiral de Cornu es el fundamento geométrico del cálculo.

## 2. Instrucciones de Uso

### a) Requisitos

Instale las dependencias necesarias:

-   `numpy`
-   `matplotlib`
-   `scipy`

Puede hacerlo fácilmente con el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### b) Ejecución

Para iniciar la simulación:
```bash
python "Punto 3.py"
```

### c) Interfaz Gráfica

La interfaz consta de:
-   **Visualización 2D (Arriba)**: Una representación del patrón de intensidad 1D, replicado verticalmente para simular el aspecto en una pantalla.
-   **Perfil de Intensidad (Abajo)**: El gráfico cuantitativo de la intensidad `I` en función de la posición `x'` en la pantalla. El título del gráfico muestra el **Número de Fresnel (F)** calculado en tiempo real.
-   **Controles (Inferior)**:
    -   **Selector de Geometría**: Para elegir entre `Rendija simple`, `Doble rendija` y `Semiplano`.
    -   **Sliders**: Permiten ajustar los parámetros `λ`, `a` (semi-ancho), `d` (semi-separación), y `z` (distancia).

## 3. Evolución del Patrón de Fresnel a Fraunhofer

Este simulador es ideal para observar la transición entre los regímenes de difracción. Para ello, fije los parámetros de la abertura (`a`, `d`) y la longitud de onda (`λ`), y luego manipule la distancia `z` o, equivalentemente, el Número de Fresnel `F`.

1.  **Número de Fresnel Grande (F > 1)**: Mueva el slider de `z` a valores pequeños.
    -   Para una **rendija simple**, observará un patrón con muchas ondulaciones dentro de la "sombra" geométrica, que se parece a la forma de la rendija.
    -   Para un **semiplano**, verá las características franjas de borde que se adentran en la zona iluminada.

2.  **Número de Fresnel Pequeño (F < 0.1)**: Mueva el slider de `z` a valores grandes.
    -   Verá que el patrón de la **rendija simple** se suaviza y converge a la forma familiar de `sinc²` del patrón de Fraunhofer.
    -   El patrón de la **doble rendija** convergerá al patrón de interferencia de Young (`cos²`) modulado por la envolvente de difracción `sinc²`.

Al variar `z`, el título del gráfico mostrará cómo cambia `F`, permitiéndole correlacionar directamente el Número de Fresnel con la forma del patrón observado, demostrando la unificación de los fenómenos de difracción de campo cercano y campo lejano.
