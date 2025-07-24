# -*- coding: utf-8 -*-
"""
Este script implementa un simulador interactivo para calcular y visualizar el
patrón de difracción de Fraunhofer de diversas aberturas. Utiliza la
Transformada Rápida de Fourier (FFT) para calcular el patrón de difracción,
lo que permite analizar cualquier forma de abertura.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.fft import fft2, fftshift, fftfreq
from matplotlib.colors import PowerNorm

class DiffractionSimulator:
    """
    Clase principal del simulador de difracción.

    Gestiona la creación de aberturas, el cálculo de la difracción mediante FFT,
    y la interfaz gráfica interactiva para la manipulación de parámetros.
    """
    def __init__(self):
        """
        Inicializa el simulador, configurando la figura de Matplotlib y los
        parámetros físicos y de simulación iniciales.
        """
        # Configuración de la ventana y los ejes de la gráfica
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.45)

        # ======================================================================
        # PARÁMETROS DE LA SIMULACIÓN Y FÍSICOS
        # ======================================================================
        self.current_scene = 'Círculo' # Abertura inicial
        self.N = 512          # Resolución de la grilla de simulación (NxN puntos)
        self.dx = 1e-5        # Tamaño de cada celda de la grilla (m)
        self.z = 1.0          # Distancia al plano de observación (m)
        self.wavelength = 550e-9 # Longitud de onda de la luz (m)
        self.zoom = 10.0      # Rango de visualización en el patrón de difracción (mm)

        # Parámetros geométricos iniciales para cada tipo de abertura
        self.circle_diameter = 1.0e-3
        self.rect_width = 0.5e-3
        self.rect_height = 2.0e-3
        self.slit_width = 0.05e-3
        self.slit_separation = 0.3e-3
        self.slit_height = 2.0e-3
        self.grid_size = 1.0e-3
        self.punto_radio = 0.1e-3
        self.flor_petalos = 6
        self.flor_rad_ext = 1.5e-3
        self.flor_rad_int = 0.5e-3

        # Variable para optimizar el renderizado durante el arrastre de sliders
        self.is_dragging = False

        # Iniciar la interfaz de usuario y el primer cálculo
        self.setup_ui()
        self.update_plot()

    def on_slider_press(self, event):
        """Callback para detectar cuándo se empieza a arrastrar un slider."""
        self.is_dragging = True

    def on_slider_release(self, event):
        """Callback para detectar cuándo se suelta un slider."""
        self.is_dragging = False
        self.update_plot() # Realiza un renderizado de alta calidad al soltar

    def create_aperture(self):
        """
        Crea la matriz 2D que representa la función de la abertura.
        La matriz contiene valores de 1 (abierto) y 0 (opaco).

        Returns:
            np.ndarray: Matriz 2D de la abertura.
        """
        # Creación de una grilla de coordenadas (x, y) en el plano de la abertura
        x = np.arange(-self.N/2, self.N/2) * self.dx
        y = np.arange(-self.N/2, self.N/2) * self.dx
        X, Y = np.meshgrid(x, y)
        
        # Selección de la función de creación de máscara según la escena actual
        if self.current_scene == 'Círculo':
            radio = self.circle_diameter / 2
            return (X**2 + Y**2 <= radio**2).astype(float)
        elif self.current_scene == 'Rectángulo':
            return ((np.abs(X) <= self.rect_width/2) & (np.abs(Y) <= self.rect_height/2)).astype(float)
        elif self.current_scene == 'Doble Rendija':
            m1 = (np.abs(X - self.slit_separation/2) <= self.slit_width/2) & (np.abs(Y) <= self.slit_height/2)
            m2 = (np.abs(X + self.slit_separation/2) <= self.slit_width/2) & (np.abs(Y) <= self.slit_height/2)
            return (m1 | m2).astype(float)
        elif self.current_scene == 'Matriz de Puntos 4x4':
            return self.create_matriz_nxn(X, Y, 4, self.grid_size, self.punto_radio)
        elif self.current_scene == 'Flor':
            return self.create_flor(X, Y)
        elif self.current_scene == 'Gato':
            # Esta es una abertura no convencional para demostrar la flexibilidad del método FFT
            return self.create_gato()
        return np.zeros((self.N, self.N))

    def create_matriz_nxn(self, X, Y, n, tamano_cuadricula, radio_punto):
        """Crea una matriz de n x n puntos circulares."""
        m = np.zeros_like(X)
        paso = tamano_cuadricula / n
        for i in np.linspace(-tamano_cuadricula/2 + paso/2, tamano_cuadricula/2 - paso/2, n):
            for j in np.linspace(-tamano_cuadricula/2 + paso/2, tamano_cuadricula/2 - paso/2, n):
                m += ((X-i)**2 + (Y-j)**2 <= radio_punto**2)
        return np.clip(m, 0, 1)

    def create_flor(self, X, Y):
        """Crea una abertura con forma de flor utilizando coordenadas polares."""
        theta = np.arctan2(Y, X) # Ángulo polar
        r = np.sqrt(X**2 + Y**2) # Radio polar
        # Ecuación de una rosa/rodonea para definir el radio en función del ángulo
        factor_petalo = np.sin(self.flor_petalos * theta)**2
        radio = self.flor_rad_int + (self.flor_rad_ext - self.flor_rad_int) * factor_petalo
        return (r <= radio).astype(float)

    def create_gato(self):
        """Crea una abertura con una silueta de gato."""
        mask = np.zeros((self.N, self.N))
        y, x = np.ogrid[-self.N/2:self.N/2, -self.N/2:self.N/2]
        
        # Cabeza (círculo)
        cabeza = x**2 + y**2 <= (self.N//4)**2
        mask[cabeza] = 1

        # Orejas (triángulos)
        y_oreja, x_oreja = np.ogrid[0:self.N//8, -self.N//16:self.N//16]
        triangulo = np.abs(x_oreja) <= y_oreja/2
        
        # Posicionamiento de las orejas
        oreja_izq_y_slice = slice(self.N//8, self.N//4)
        oreja_izq_x_slice = slice(self.N//4, 3*self.N//8)
        oreja_der_y_slice = slice(self.N//8, self.N//4)
        oreja_der_x_slice = slice(5*self.N//8, 3*self.N//4)
        
        # Es necesario rotar el triángulo para que apunte hacia arriba
        triangulo_rotado = np.flipud(triangulo)
        
        mask[oreja_izq_y_slice, oreja_izq_x_slice] = triangulo_rotado
        mask[oreja_der_y_slice, oreja_der_x_slice] = triangulo_rotado

        return mask

    def calculate_diffraction(self, aperture):
        """
        Calcula el patrón de difracción usando la Transformada de Fourier 2D.

        El Teorema de Difracción de Fraunhofer establece que el campo lejano es
        la Transformada de Fourier de la función de la abertura.

        Args:
            aperture (np.ndarray): La matriz 2D de la abertura.

        Returns:
            tuple: (intensidad, coordenadas_x_observacion, coordenadas_y_observacion)
        """
        # 1. Calcular la FFT 2D de la abertura. fftshift centra la frecuencia cero.
        campo_fft = fftshift(fft2(aperture)) * self.dx**2

        # 2. Calcular las frecuencias espaciales correspondientes a la FFT.
        fx = fftshift(fftfreq(self.N, self.dx))
        fy = fftshift(fftfreq(self.N, self.dx))

        # 3. Convertir frecuencias espaciales (fx, fy) a coordenadas en el plano
        #    de observación (x', y') usando la relación de Fraunhofer.
        x_obs = self.wavelength * self.z * fx
        y_obs = self.wavelength * self.z * fy

        # 4. La intensidad es el módulo al cuadrado del campo complejo.
        intensidad = np.abs(campo_fft)**2

        # 5. Normalizar la intensidad para que el máximo sea 1.
        return intensidad / np.max(intensidad), x_obs, y_obs

    def setup_ui(self):
        """Configura todos los widgets de la interfaz gráfica."""
        self.axcolor = 'lightgoldenrodyellow'
        
        # Botones de radio para seleccionar la abertura
        self.ax_radio = plt.axes([0.05, 0.05, 0.15, 0.25], facecolor=self.axcolor)
        self.radio_buttons = RadioButtons(self.ax_radio, ('Círculo', 'Rectángulo', 'Doble Rendija', 'Matriz de Puntos 4x4', 'Flor', 'Gato'), active=0)
        self.radio_buttons.on_clicked(self.set_scene)

        # Sliders para parámetros globales
        self.ax_wavelength = plt.axes([0.3, 0.3, 0.6, 0.03], facecolor=self.axcolor)
        self.slider_wavelength = Slider(self.ax_wavelength, 'Long. de Onda (nm)', 400, 700, valinit=self.wavelength * 1e9)
        self.ax_z = plt.axes([0.3, 0.25, 0.6, 0.03], facecolor=self.axcolor)
        self.slider_z = Slider(self.ax_z, 'Distancia z (m)', 0.1, 10.0, valinit=self.z)
        self.ax_zoom = plt.axes([0.3, 0.2, 0.6, 0.03], facecolor=self.axcolor)
        self.slider_zoom = Slider(self.ax_zoom, 'Zoom (mm)', 1, 100, valinit=self.zoom)

        # Diccionario de sliders específicos para cada abertura
        self.sliders = {
            'Círculo': [Slider(plt.axes([0.3, 0.15, 0.6, 0.03], facecolor=self.axcolor), 'Diámetro (mm)', 0.1, 5, valinit=self.circle_diameter * 1e3)],
            'Rectángulo': [Slider(plt.axes([0.3, 0.15, 0.6, 0.03], facecolor=self.axcolor), 'Ancho (mm)', 0.1, 5, valinit=self.rect_width * 1e3),
                           Slider(plt.axes([0.3, 0.1, 0.6, 0.03], facecolor=self.axcolor), 'Alto (mm)', 0.1, 5, valinit=self.rect_height * 1e3)],
            'Doble Rendija': [Slider(plt.axes([0.3, 0.15, 0.6, 0.03], facecolor=self.axcolor), 'Ancho (mm)', 0.01, 0.5, valinit=self.slit_width * 1e3),
                              Slider(plt.axes([0.3, 0.1, 0.6, 0.03], facecolor=self.axcolor), 'Separación (mm)', 0.1, 2, valinit=self.slit_separation * 1e3)],
            'Matriz de Puntos 4x4': [Slider(plt.axes([0.3, 0.15, 0.6, 0.03], facecolor=self.axcolor), 'Tamaño Cuadrícula (mm)', 0.1, 3, valinit=self.grid_size * 1e3),
                                     Slider(plt.axes([0.3, 0.1, 0.6, 0.03], facecolor=self.axcolor), 'Radio Punto (mm)', 0.01, 0.5, valinit=self.punto_radio * 1e3)],
            'Flor': [Slider(plt.axes([0.3, 0.15, 0.6, 0.03], facecolor=self.axcolor), 'Pétalos', 3, 20, valinit=self.flor_petalos, valstep=1),
                     Slider(plt.axes([0.3, 0.1, 0.6, 0.03], facecolor=self.axcolor), 'Radio Ext (mm)', 0.1, 5, valinit=self.flor_rad_ext * 1e3),
                     Slider(plt.axes([0.3, 0.05, 0.6, 0.03], facecolor=self.axcolor), 'Radio Int (mm)', 0.1, 5, valinit=self.flor_rad_int * 1e3)],
            'Gato': []
        }

        # Conectar todos los sliders a sus funciones de callback
        self.slider_wavelength.on_changed(self.update_params)
        self.slider_z.on_changed(self.update_params)
        self.slider_zoom.on_changed(self.update_params)

        for scene_sliders in self.sliders.values():
            for slider in scene_sliders:
                slider.on_changed(self.update_params)
                slider.ax.set_visible(False)
                self.fig.canvas.mpl_connect('button_press_event', self.on_slider_press)
                self.fig.canvas.mpl_connect('button_release_event', self.on_slider_release)
        
        self.set_scene(self.current_scene)

    def set_scene(self, label):
        """Función que se llama al cambiar de tipo de abertura. Muestra/oculta los sliders correspondientes."""
        self.current_scene = label
        for scene, scene_sliders in self.sliders.items():
            for slider in scene_sliders:
                slider.ax.set_visible(scene == label)
        self.update_plot()

    def update_params(self, val):
        """Actualiza los parámetros desde los sliders y lanza el redibujado."""
        self.wavelength = self.slider_wavelength.val * 1e-9
        self.z = self.slider_z.val
        self.zoom = self.slider_zoom.val

        if self.current_scene == 'Círculo':
            self.circle_diameter = self.sliders['Círculo'][0].val * 1e-3
        elif self.current_scene == 'Rectángulo':
            self.rect_width = self.sliders['Rectángulo'][0].val * 1e-3
            self.rect_height = self.sliders['Rectángulo'][1].val * 1e-3
        elif self.current_scene == 'Doble Rendija':
            self.slit_width = self.sliders['Doble Rendija'][0].val * 1e-3
            self.slit_separation = self.sliders['Doble Rendija'][1].val * 1e-3
        elif self.current_scene == 'Matriz de Puntos 4x4':
            self.grid_size = self.sliders['Matriz de Puntos 4x4'][0].val * 1e-3
            self.punto_radio = self.sliders['Matriz de Puntos 4x4'][1].val * 1e-3
        elif self.current_scene == 'Flor':
            self.flor_petalos = self.sliders['Flor'][0].val
            self.flor_rad_ext = self.sliders['Flor'][1].val * 1e-3
            self.flor_rad_int = self.sliders['Flor'][2].val * 1e-3
            
        if not self.is_dragging:
            self.update_plot()

    def update_plot(self, event=None):
        """
        Función principal de actualización. Crea la abertura, calcula la difracción
        y actualiza los tres gráficos: abertura, patrón 2D y perfil de intensidad.
        """
        # 1. Crear la abertura
        aperture = self.create_aperture()

        # 2. Calcular el patrón de difracción
        intensity, x_obs, y_obs = self.calculate_diffraction(aperture)

        # 3. Actualizar el gráfico de la abertura (ax1)
        extent_apertura = [-self.N/2*self.dx*1e3, self.N/2*self.dx*1e3, -self.N/2*self.dx*1e3, self.N/2*self.dx*1e3]
        self.ax1.clear()
        self.ax1.imshow(aperture, cmap='gray', extent=extent_apertura, origin='lower')
        self.ax1.set_title('Abertura')
        self.ax1.set_xlabel('x (mm)')
        self.ax1.set_ylabel('y (mm)')
        self.ax1.grid(True, alpha=0.3)

        # 4. Actualizar el gráfico del patrón 2D (ax2)
        zoom_m = self.zoom * 1e-3
        extent_difraccion = [-zoom_m*1e3, zoom_m*1e3, -zoom_m*1e3, zoom_m*1e3]
        
        # Recortar la región de interés según el zoom
        idx_x = np.abs(x_obs) <= zoom_m
        idx_y = np.abs(y_obs) <= zoom_m
        I_zoom = intensity[np.outer(idx_y, idx_x)].reshape((np.sum(idx_y), np.sum(idx_x)))
        
        # Usar interpolación 'none' mientras se arrastra para mayor fluidez
        interpolation_method = 'bicubic' if not self.is_dragging else 'none'

        self.ax2.clear()
        # Se usa PowerNorm para realzar los detalles débiles (similar a una escala log)
        self.ax2.imshow(I_zoom, cmap='inferno', norm=PowerNorm(gamma=0.3), extent=extent_difraccion, interpolation=interpolation_method, origin='lower')
        self.ax2.set_title('Patrón de Difracción (FFT)')
        self.ax2.set_xlabel("x' (mm)")
        self.ax2.set_ylabel("y' (mm)")
        self.ax2.grid(True, alpha=0.3)

        # 5. Actualizar el gráfico del perfil de intensidad (ax3)
        perfil = I_zoom[I_zoom.shape[0] // 2, :]
        x_perfil = np.linspace(-zoom_m*1e3, zoom_m*1e3, len(perfil))
        self.ax3.clear()
        self.ax3.plot(x_perfil, perfil)
        self.ax3.set_title('Perfil de Intensidad Central')
        self.ax3.set_xlabel("x' (mm)")
        self.ax3.set_ylabel('Intensidad Normalizada')
        self.ax3.grid(True)
        self.ax3.set_ylim(0, 1.1)

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    sim = DiffractionSimulator()
    plt.show()
