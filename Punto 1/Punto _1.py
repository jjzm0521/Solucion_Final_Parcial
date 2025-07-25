# -*- coding: utf-8 -*-
"""
Análisis de la difracción de Fraunhofer para dos aberturas: un rectángulo y un anillo,
separados por una distancia D. Este script calcula y visualiza el patrón de
difracción resultante, permitiendo la variación interactiva de los parámetros físicos.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from scipy.special import j1
import matplotlib.colors as mcolors

class FraunhoferDiffraction:
    """
    Clase principal para la simulación de la difracción de Fraunhofer.

    Esta clase maneja la configuración de los parámetros, el cálculo de la intensidad
    del patrón de difracción, y la creación de la interfaz gráfica interactiva.
    """
    def __init__(self):
        """
        Inicializa la simulación. Define los parámetros físicos iniciales del sistema
        y configura la ventana de la interfaz gráfica.
        """
        # ======================================================================
        # PARÁMETROS FÍSICOS DEL SISTEMA
        # ======================================================================
        self.params = {
            'lambda': 550e-9,      # Longitud de onda de la luz (m)
            'D': 2e-3,             # Distancia entre los centros de las aberturas (m)
            'a': 0.5e-3,           # Ancho del rectángulo (m)
            'b': 0.5e-3,           # Alto del rectángulo (m)
            'R1': 0.2e-3,          # Radio interno del anillo (m)
            'R2': 0.8e-3,          # Radio externo del anillo (m)
            'z': 1,                # Distancia del plano de la abertura al plano de observación (m)
            'n': 1,                # Índice de refracción del medio (adimensional)
            'I_source': 1,         # Intensidad de la fuente de luz (unidades arbitrarias)
            'range': 5e-3,         # Rango de visualización en el plano de observación (m)
            'aperture_range': 2e-3 # Rango de visualización en el plano de la abertura (m)
        }
        
        # ======================================================================
        # CONFIGURACIÓN DE LA SIMULACIÓN
        # ======================================================================
        self.grid_size = 400          # Resolución de la grilla para el patrón de difracción
        self.aperture_grid_size = 200 # Resolución de la grilla para la visualización de la abertura

        # Iniciar la configuración de la interfaz gráfica
        self.setup_plot()
    
    def wavelength_to_rgb(self, wavelength_nm):
        """
        Convierte una longitud de onda en nanómetros a un color RGB aproximado.
        Esto permite visualizar el color de la luz utilizada en la simulación.

        Args:
            wavelength_nm (float): Longitud de onda en nanómetros.

        Returns:
            tuple: Un triplete (R, G, B) con valores entre 0 y 1.
        """
        wavelength = wavelength_nm
        
        if 380 <= wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wavelength < 510:
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wavelength < 645:
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        elif 645 <= wavelength <= 780:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            r, g, b = 0.0, 0.0, 0.0 # Fuera del espectro visible

        # Ajuste de brillo para que los extremos del espectro no sean tan oscuros
        if 380 <= wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 645 < wavelength <= 780:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
        else:
            factor = 1.0
        
        return (r * factor, g * factor, b * factor)
    
    def create_wavelength_colormap(self, wavelength_nm):
        """
        Crea un mapa de colores (colormap) personalizado que va del negro al color
        de la longitud de onda especificada.
        """
        base_color = self.wavelength_to_rgb(wavelength_nm)
        return mcolors.LinearSegmentedColormap.from_list(
            "custom_map", [(0, 0, 0), base_color]
        )
    
    def sinc(self, x):
        """
        Calcula la función sinc(x) = sin(x)/x.
        Se maneja el caso especial en x=0, donde sinc(0)=1.
        """
        return np.where(np.abs(x) < 1e-10, 1, np.sin(x) / x)
    
    def bessel_j1_normalized(self, x):
        """
        Calcula la función de Bessel de primer orden normalizada: J1(x)/x.
        Esta función aparece en la difracción de aberturas circulares.
        Se maneja el caso especial en x=0, donde el límite es 0.5.
        """
        return np.where(np.abs(x) < 1e-10, 0.5, j1(x) / x)
    
    def calculate_intensity(self, x, y):
        """
        Calcula la intensidad del patrón de difracción de Fraunhofer.

        La intensidad total es el módulo al cuadrado de la suma de los campos eléctricos
        producidos por cada abertura.

        I = |E_rect + E_anillo|^2 = |E_rect|^2 + |E_anillo|^2 + 2 * Re(E_rect * E_anillo^*)

        Args:
            x, y (np.ndarray): Coordenadas en el plano de observación.

        Returns:
            np.ndarray: La intensidad calculada en cada punto (x, y).
        """
        # ======================================================================
        # EXTRACCIÓN DE PARÁMETROS
        # ======================================================================
        lambda_val = self.params['lambda']
        D = self.params['D']
        a = self.params['a']
        b = self.params['b']
        R1 = self.params['R1']
        R2 = self.params['R2']
        z = self.params['z']
        n = self.params['n']
        I_source = self.params['I_source']
        
        # ======================================================================
        # CÁLCULOS PREVIOS
        # ======================================================================
        k = 2 * np.pi / lambda_val  # Número de onda
        r = np.sqrt(x**2 + y**2)    # Coordenada radial en el plano de observación
        
        # ======================================================================
        # CAMPO DE LA ABERTURA RECTANGULAR (centrada en y = D/2)
        # ======================================================================
        # El campo de un rectángulo es el producto de dos funciones sinc.
        beta_x = k * a * x / z
        beta_y = k * b * y / z
        
        sinc_x = self.sinc(beta_x / 2)
        sinc_y = self.sinc(beta_y / 2)
        
        # Amplitud del campo del rectángulo (sin la fase)
        rect_field_term = (b * a) * sinc_x * sinc_y

        # ======================================================================
        # CAMPO DE LA ABERTURA ANULAR (centrada en y = -D/2)
        # ======================================================================
        # El campo de un anillo es la resta de los campos de dos círculos.
        # El campo de un círculo involucra la función de Bessel J1.
        kr = k * r
        kr_safe = np.where(r > 1e-10, kr, 1e-10) # Evitar división por cero
        
        # Término de Bessel para el radio interno (R1)
        bessel_R1 = np.where(R1 > 0, R1**2 * self.bessel_j1_normalized(kr_safe * R1 / z), 0)
        # Término de Bessel para el radio externo (R2)
        bessel_R2 = np.where(R2 > 0, R2**2 * self.bessel_j1_normalized(kr_safe * R2 / z), 0)
        
        # Amplitud del campo del anillo
        ring_field_term = (2 * np.pi) * (bessel_R2 - bessel_R1) # Se simplifica el 2J1(x)/x
        
        # ======================================================================
        # CÁLCULO DE LA INTENSIDAD TOTAL
        # ======================================================================
        # La intensidad total I = |E_total|^2.
        # E_total = E_rect * exp(i*phi_rect) + E_anillo * exp(i*phi_anillo)
        # La diferencia de fase se debe a la separación D.
        
        # Término de difracción del rectángulo (I_rect = |E_rect|^2)
        term1 = rect_field_term**2
        
        # Término de difracción del anillo (I_anillo = |E_anillo|^2)
        term2 = ring_field_term**2

        # Término de interferencia
        phase = k * D * y / z # Diferencia de fase por la separación en 'y'
        term3 = 2 * rect_field_term * ring_field_term * np.cos(phase)
        
        # Constante de proporcionalidad y suma de términos
        pre_factor = I_source * (n**2 / (lambda_val**2 * z**2))
        intensity = pre_factor * (term1 + term2 + term3)
        
        return np.maximum(0, intensity)

    def calculate_pattern(self):
        """
        Genera la grilla de coordenadas y calcula el patrón de intensidad 2D.
        """
        range_val = self.params['range']
        x = np.linspace(-range_val, range_val, self.grid_size)
        y = np.linspace(-range_val, range_val, self.grid_size)
        X, Y = np.meshgrid(x, y)
        intensity = self.calculate_intensity(X, Y)
        return X, Y, intensity
    
    def create_aperture_pattern(self):
        """
        Crea una representación visual de las aberturas en su plano.
        """
        range_val = self.params['aperture_range']
        x = np.linspace(-range_val, range_val, self.aperture_grid_size)
        y = np.linspace(-range_val, range_val, self.aperture_grid_size)
        X, Y = np.meshgrid(x, y)
        
        aperture = np.zeros_like(X)
        
        # Rectángulo centrado en (0, D/2)
        rect_mask = (np.abs(X) <= self.params['a'] / 2) & \
                    (np.abs(Y - self.params['D'] / 2) <= self.params['b'] / 2)
        aperture[rect_mask] = 1
        
        # Anillo centrado en (0, -D/2)
        r_aperture = np.sqrt(X**2 + (Y + self.params['D'] / 2)**2)
        ring_mask = (r_aperture >= self.params['R1']) & (r_aperture <= self.params['R2'])
        aperture[ring_mask] = 1
        
        return X, Y, aperture
    
    def setup_plot(self):
        """
        Configura la interfaz gráfica, incluyendo la figura, los ejes y los sliders
        para el control interactivo de los parámetros.
        """
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.fig.suptitle("Simulador de Difracción de Fraunhofer", fontsize=16, fontweight='bold')

        # Layout principal con GridSpec para un mejor control
        gs_main = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.08)
        self.ax_main = self.fig.add_subplot(gs_main[0, 1]) # Eje para el patrón de difracción
        gs_left = gs_main[0, 0].subgridspec(13, 1, hspace=2.5) # Sub-layout para controles y apertura
        self.ax_aperture = self.fig.add_subplot(gs_left[0:4, 0]) # Eje para la visualización de la abertura

        # Definición de los sliders
        self.sliders = {}
        slider_defs = {
            'lambda': {'label': 'λ (nm)', 'range': (400, 700), 'format': '%.0f'},
            'D': {'label': 'D (mm)', 'range': (0.5, 5), 'format': '%.2f'},
            'a': {'label': 'a (mm)', 'range': (0, 2), 'format': '%.2f'},
            'b': {'label': 'b (mm)', 'range': (0, 2), 'format': '%.2f'},
            'R1': {'label': 'R₁ (mm)', 'range': (0, 1), 'format': '%.2f'},
            'R2': {'label': 'R₂ (mm)', 'range': (0, 2), 'format': '%.2f'},
            'z': {'label': 'z (m)', 'range': (0.5, 3), 'format': '%.1f'},
            'range': {'label': 'Rango (mm)', 'range': (2, 20), 'format': '%.1f'}
        }
        
        # Factores de conversión de unidades para los sliders
        conversions = {'lambda': 1e-9, 'D': 1e-3, 'a': 1e-3, 'b': 1e-3, 'R1': 1e-3, 'R2': 1e-3, 'z': 1, 'range': 1e-3}

        # Creación de los sliders
        for i, (key, sdef) in enumerate(slider_defs.items()):
            ax = self.fig.add_subplot(gs_left[5+i])
            val_init = self.params[key] / conversions[key]
            self.sliders[key] = Slider(ax, sdef['label'], sdef['range'][0], sdef['range'][1], valinit=val_init, valfmt=sdef['format'])
        
        for slider in self.sliders.values():
            slider.on_changed(self.update_params)
        
        self.update_plot()
    
    def update_params(self, val):
        """
        Función callback que se ejecuta cuando un slider cambia. Actualiza los
        parámetros y redibuja los gráficos.
        """
        conversions = {'lambda': 1e-9, 'D': 1e-3, 'a': 1e-3, 'b': 1e-3, 'R1': 1e-3, 'R2': 1e-3, 'z': 1, 'range': 1e-3}
        for key, slider in self.sliders.items():
            self.params[key] = slider.val * conversions[key]
        
        # Asegura que el radio interno no sea mayor que el externo
        if self.params['R1'] > self.params['R2']:
            self.params['R1'] = self.params['R2']
            self.sliders['R1'].set_val(self.params['R1'] / conversions['R1'])

        self.update_plot()
    
    def update_plot(self):
        """
        Actualiza y redibuja tanto el gráfico de la abertura como el del patrón de difracción.
        """
        self.ax_main.clear()
        self.ax_aperture.clear()
        
        # Dibuja la abertura
        X_ap, Y_ap, aperture = self.create_aperture_pattern()
        range_mm_ap = self.params['aperture_range'] * 1e3
        self.ax_aperture.imshow(aperture, extent=[-range_mm_ap, range_mm_ap, -range_mm_ap, range_mm_ap],
                                cmap='gray', origin='lower')
        self.ax_aperture.set_title('Apertura Original', fontsize=12, fontweight='bold')
        self.ax_aperture.set_xlabel('x (mm)')
        self.ax_aperture.set_ylabel('y (mm)')
        self.ax_aperture.grid(True, linestyle='--', alpha=0.4)
        self.ax_aperture.set_aspect('equal', 'box')

        # Calcula y dibuja el patrón de difracción
        X, Y, intensity = self.calculate_pattern()
        
        # Se aplica una escala logarítmica para mejorar la visualización de los
        # máximos secundarios, que son mucho menos intensos que el central.
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity_log = np.log1p(intensity / max_intensity * 1000)
            intensity_norm = intensity_log / np.max(intensity_log)
        else:
            intensity_norm = intensity

        wavelength_nm = self.params['lambda'] * 1e9
        custom_cmap = self.create_wavelength_colormap(wavelength_nm)
        
        range_mm = self.params['range'] * 1e3
        self.ax_main.imshow(intensity_norm, 
                            extent=[-range_mm, range_mm, -range_mm, range_mm],
                            cmap=custom_cmap, origin='lower', interpolation='bilinear')
        
        self.ax_main.set_title('Patrón de Difracción de Fraunhofer', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel("x' (mm)", fontsize=12)
        self.ax_main.set_ylabel("y' (mm)", fontsize=12)
        
        # Añade texto informativo sobre la longitud de onda y los parámetros
        wavelength_color = self.wavelength_to_rgb(wavelength_nm)
        info_text = f'λ = {wavelength_nm:.0f} nm'
        self.ax_main.text(0.98, 0.98, info_text, transform=self.ax_main.transAxes,
                          fontsize=12, fontweight='bold', color='white', ha='right', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=wavelength_color, alpha=0.8))
        
        params_text = (f"D = {self.params['D']*1e3:.2f} mm\n"
                       f"a = {self.params['a']*1e3:.2f} mm, b = {self.params['b']*1e3:.2f} mm\n"
                       f"R₁ = {self.params['R1']*1e3:.2f} mm, R₂ = {self.params['R2']*1e3:.2f} mm\n"
                       f"z = {self.params['z']:.1f} m")
        self.ax_main.text(0.02, 0.02, params_text, transform=self.ax_main.transAxes,
                          fontsize=10, color='white', verticalalignment='bottom',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Ejecuta la simulación y muestra la ventana de Matplotlib."""
        plt.show()

def main():
    """Función principal que instancia y ejecuta el simulador."""
    print("=" * 60)
    print("      SIMULADOR DE DIFRACCIÓN DE FRAUNHOFER (RECTÁNGULO + ANILLO)")
    print("=" * 60)
    print("\nInstrucciones:")
    print("- Use los controles deslizantes para ajustar los parámetros del sistema.")
    print("- La visualización de la abertura y el patrón de difracción se actualizarán en tiempo real.")
    print("\n🚀 ¡Disfruta explorando la difracción de Fraunhofer!")
    print("=" * 60)
    
    simulator = FraunhoferDiffraction()
    simulator.run()

if __name__ == "__main__":
    main()