
# -*- coding: utf-8 -*-
"""
===================================================================
SIMULADOR INTERACTIVO DE DIFRACCIÓN 2D CON INTERFAZ GRÁFICA
===================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.fft import fft2, fftshift, fftfreq
from matplotlib.colors import PowerNorm
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Configuración global para mejor calidad
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['text.color'] = 'black'

class SimuladorDifraccionInteractivo:
    """
    Simulador interactivo de difracción con interfaz gráfica completa.
    """
    
    def __init__(self):
        """Inicializa el simulador con parámetros por defecto."""
        # Parámetros físicos
        self.N = 256  # Reducido para mejor rendimiento interactivo
        self.dx = 5e-6  # 5 μm
        self.z = 1.0  # 1 metro
        self.wavelength = 550e-9  # 550 nm (verde)
        
        # Parámetros de abertura
        self.apertura_tipo = 'Círculo'
        
        # Parámetros para cada tipo de abertura
        self.params = {
            'circle_diameter': 0.5e-3,      # 0.5 mm
            'circle_eccentricity': 0.0,     # Sin excentricidad
            'rect_width': 0.5e-3,           # 0.5 mm
            'rect_height': 1.0e-3,          # 1 mm
            'slit_width': 0.05e-3,          # 0.05 mm
            'slit_separation': 0.3e-3,      # 0.3 mm
            'slit_height': 2.0e-3,          # 2 mm
            'grid_diameter': 0.1e-3,        # 0.1 mm
            'grid_spacing': 0.4e-3,         # 0.4 mm
            'grid_disorder': 0.0,           # Sin desorden
            'grid_nx': 4,                   # 4x4 rejilla
            'grid_ny': 4,
        }
        
        self.zoom = 10.0  # mm
        self.colormap = 'viridis'
        
        # Variables para la interfaz
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.im_apertura = None
        self.im_difraccion = None
        self.line_perfil = None
        
    def wavelength_to_color(self, wavelength_nm):
        """Convierte longitud de onda a color RGB."""
        if wavelength_nm < 440:
            return '#8B00FF'  # Violeta
        elif wavelength_nm < 490:
            return '#0000FF'  # Azul
        elif wavelength_nm < 510:
            return '#00FFFF'  # Cian
        elif wavelength_nm < 580:
            return '#00FF00'  # Verde
        elif wavelength_nm < 645:
            return '#FFFF00'  # Amarillo
        else:
            return '#FF0000'  # Rojo
    
    def crear_mascara_abertura(self):
        """Crea la máscara de la abertura según el tipo seleccionado."""
        # Crear grilla de coordenadas
        x = np.arange(-self.N//2, self.N//2) * self.dx
        y = np.arange(-self.N//2, self.N//2) * self.dx
        X, Y = np.meshgrid(x, y)
        
        mascara = np.zeros((self.N, self.N))
        
        if self.apertura_tipo == 'Círculo':
            radio = self.params['circle_diameter'] / 2
            ecc = self.params['circle_eccentricity']
            # Aplicar excentricidad
            a = radio
            b = radio * np.sqrt(1 - ecc**2) if ecc < 1 else radio
            mascara = ((X/a)**2 + (Y/b)**2 <= 1).astype(float)
            
        elif self.apertura_tipo == 'Rectángulo':
            ancho = self.params['rect_width']
            alto = self.params['rect_height']
            mascara = (np.abs(X) <= ancho/2) & (np.abs(Y) <= alto/2)
            mascara = mascara.astype(float)
            
        elif self.apertura_tipo == 'Doble Rendija':
            ancho = self.params['slit_width']
            sep = self.params['slit_separation']
            alto = self.params['slit_height']
            
            mascara1 = (np.abs(X - sep/2) <= ancho/2) & (np.abs(Y) <= alto/2)
            mascara2 = (np.abs(X + sep/2) <= ancho/2) & (np.abs(Y) <= alto/2)
            mascara = (mascara1 | mascara2).astype(float)
            
        elif self.apertura_tipo == 'Rejilla':
            radio = self.params['grid_diameter'] / 2
            espaciado = self.params['grid_spacing']
            desorden = self.params['grid_disorder']
            nx = int(self.params['grid_nx'])
            ny = int(self.params['grid_ny'])
            
            # Crear rejilla con posible desorden
            for i in range(-(nx//2), (nx//2)+1):
                for j in range(-(ny//2), (ny//2)+1):
                    # Posición base
                    x_centro = i * espaciado
                    y_centro = j * espaciado
                    
                    # Añadir desorden aleatorio
                    if desorden > 0:
                        x_centro += np.random.uniform(-desorden*espaciado/2, desorden*espaciado/2)
                        y_centro += np.random.uniform(-desorden*espaciado/2, desorden*espaciado/2)
                    
                    mascara_temp = ((X - x_centro)**2 + (Y - y_centro)**2 <= radio**2)
                    mascara = np.maximum(mascara, mascara_temp)
                    
        elif self.apertura_tipo == 'Círculo + Cuadrado':
            # Círculo
            radio = self.params['circle_diameter'] / 2
            mascara_circulo = (X**2 + Y**2 <= radio**2)
            
            # Cuadrado
            lado = self.params['rect_width']
            mascara_cuadrado = (np.abs(X) <= lado/2) & (np.abs(Y) <= lado/2)
            
            mascara = np.maximum(mascara_circulo, mascara_cuadrado).astype(float)
            
        return mascara
    
    def calcular_difraccion(self, mascara):
        """Calcula el patrón de difracción."""
        # FFT de la abertura
        campo_fft = fftshift(fft2(mascara)) * self.dx**2
        
        # Frecuencias espaciales
        fx = fftshift(fftfreq(self.N, self.dx))
        fy = fftshift(fftfreq(self.N, self.dx))
        
        # Coordenadas en el plano de observación
        x_obs = self.wavelength * self.z * fx
        y_obs = self.wavelength * self.z * fy
        
        # Intensidad
        intensidad = np.abs(campo_fft)**2
        
        # Normalizar
        if np.max(intensidad) > 0:
            intensidad = intensidad / np.max(intensidad)
            
        return intensidad, x_obs, y_obs
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica completa."""
        # Crear figura principal
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Simulador Interactivo de Difracción de Fraunhofer', fontsize=16, fontweight='bold')
        
        # Definir layout
        # Área principal para visualización
        self.axes['laser'] = plt.subplot2grid((10, 12), (0, 0), rowspan=5, colspan=2)
        self.axes['apertura'] = plt.subplot2grid((10, 12), (0, 2), rowspan=5, colspan=3)
        self.axes['pantalla'] = plt.subplot2grid((10, 12), (0, 5), rowspan=5, colspan=3)
        self.axes['difraccion'] = plt.subplot2grid((10, 12), (0, 8), rowspan=5, colspan=4)
        self.axes['perfil'] = plt.subplot2grid((10, 12), (5, 8), rowspan=3, colspan=4)
        
        # Área para controles
        self.axes['controles'] = plt.subplot2grid((10, 12), (6, 0), rowspan=4, colspan=8)
        self.axes['controles'].axis('off')
        
        # Configurar visualización del láser
        self.axes['laser'].set_xlim(0, 1)
        self.axes['laser'].set_ylim(0, 1)
        self.axes['laser'].axis('off')
        
        # Dibujar láser
        laser_body = Rectangle((0.2, 0.4), 0.6, 0.2, facecolor='gray', edgecolor='black')
        self.axes['laser'].add_patch(laser_body)
        laser_lens = Circle((0.8, 0.5), 0.1, facecolor='red', edgecolor='darkred')
        self.axes['laser'].add_patch(laser_lens)
        self.axes['laser'].text(0.5, 0.2, 'Láser', ha='center', fontsize=12)
        
        # Dibujar haz de luz (se actualizará con el color)
        self.laser_beam = None
        
        # Configurar visualización de apertura
        self.axes['apertura'].set_title('Apertura', fontsize=14, fontweight='bold')
        self.axes['apertura'].set_xlabel('x (mm)')
        self.axes['apertura'].set_ylabel('y (mm)')
        
        # Configurar visualización de pantalla
        self.axes['pantalla'].set_xlim(0, 1)
        self.axes['pantalla'].set_ylim(0, 1)
        self.axes['pantalla'].axis('off')
        
        # Dibujar pantalla
        screen = Rectangle((0.8, 0.1), 0.1, 0.8, facecolor='black', edgecolor='gray')
        self.axes['pantalla'].add_patch(screen)
        self.axes['pantalla'].text(0.5, 0.05, 'Pantalla', ha='center', fontsize=12)
        
        # Configurar visualización del patrón
        self.axes['difraccion'].set_title('Patrón de Difracción', fontsize=14, fontweight='bold')
        self.axes['difraccion'].set_xlabel("x' (mm)")
        self.axes['difraccion'].set_ylabel("y' (mm)")
        
        # Configurar perfil
        self.axes['perfil'].set_title('Perfil de Intensidad Central', fontsize=12)
        self.axes['perfil'].set_xlabel("x' (mm)")
        self.axes['perfil'].set_ylabel('Intensidad')
        self.axes['perfil'].grid(True, alpha=0.3)
        
        # Crear controles
        self.crear_controles()
        
        # Actualización inicial
        self.actualizar(None)
        
        plt.tight_layout()
        
    def crear_controles(self):
        """Crea todos los controles interactivos."""
        # Control de longitud de onda
        ax_wavelength = plt.axes([0.15, 0.45, 0.3, 0.03])
        self.sliders['wavelength'] = Slider(
            ax_wavelength, 'λ (nm)', 380, 780, 
            valinit=self.wavelength*1e9, valstep=10
        )
        self.sliders['wavelength'].on_changed(self.actualizar)
        
        # Barra de colores para longitud de onda
        ax_colorbar = plt.axes([0.15, 0.49, 0.3, 0.02])
        ax_colorbar.axis('off')
        gradient = np.linspace(380, 780, 256).reshape(1, -1)
        colors = [self.wavelength_to_color(wl) for wl in gradient[0]]
        for i in range(len(colors)-1):
            ax_colorbar.axhspan(0, 1, xmin=i/255, xmax=(i+1)/255, 
                              facecolor=colors[i])
        
        # Control de distancia
        ax_distance = plt.axes([0.15, 0.38, 0.3, 0.03])
        self.sliders['distance'] = Slider(
            ax_distance, 'z (m)', 0.1, 5.0, 
            valinit=self.z, valstep=0.1
        )
        self.sliders['distance'].on_changed(self.actualizar)
        
        # Control de zoom
        ax_zoom = plt.axes([0.15, 0.31, 0.3, 0.03])
        self.sliders['zoom'] = Slider(
            ax_zoom, 'Zoom (mm)', 1, 50, 
            valinit=self.zoom, valstep=1
        )
        self.sliders['zoom'].on_changed(self.actualizar)
        
        # Selector de tipo de abertura
        ax_radio = plt.axes([0.5, 0.25, 0.15, 0.25])
        self.radio = RadioButtons(
            ax_radio, 
            ('Círculo', 'Rectángulo', 'Doble Rendija', 'Rejilla', 'Círculo + Cuadrado'),
            active=0
        )
        self.radio.on_clicked(self.cambiar_abertura)
        
        # Controles específicos por abertura
        self.crear_controles_abertura()
        
    def crear_controles_abertura(self):
        """Crea controles específicos para cada tipo de abertura."""
        # Limpiar controles previos específicos
        for key in list(self.sliders.keys()):
            if key not in ['wavelength', 'distance', 'zoom']:
                self.sliders[key].ax.remove()
                del self.sliders[key]
        
        y_pos = 0.22
        
        if self.apertura_tipo == 'Círculo':
            # Diámetro
            ax = plt.axes([0.15, y_pos, 0.3, 0.03])
            self.sliders['circle_diameter'] = Slider(
                ax, 'Diámetro (mm)', 0.1, 3.0, 
                valinit=self.params['circle_diameter']*1e3, valstep=0.05
            )
            self.sliders['circle_diameter'].on_changed(self.actualizar)
            
            # Excentricidad
            ax = plt.axes([0.15, y_pos-0.05, 0.3, 0.03])
            self.sliders['circle_eccentricity'] = Slider(
                ax, 'Excentricidad', 0.0, 0.99, 
                valinit=self.params['circle_eccentricity'], valstep=0.05
            )
            self.sliders['circle_eccentricity'].on_changed(self.actualizar)
            
        elif self.apertura_tipo == 'Rectángulo':
            # Ancho
            ax = plt.axes([0.15, y_pos, 0.3, 0.03])
            self.sliders['rect_width'] = Slider(
                ax, 'Ancho (mm)', 0.1, 3.0, 
                valinit=self.params['rect_width']*1e3, valstep=0.05
            )
            self.sliders['rect_width'].on_changed(self.actualizar)
            
            # Alto
            ax = plt.axes([0.15, y_pos-0.05, 0.3, 0.03])
            self.sliders['rect_height'] = Slider(
                ax, 'Alto (mm)', 0.1, 3.0, 
                valinit=self.params['rect_height']*1e3, valstep=0.05
            )
            self.sliders['rect_height'].on_changed(self.actualizar)
            
        elif self.apertura_tipo == 'Doble Rendija':
            # Ancho de rendija
            ax = plt.axes([0.15, y_pos, 0.3, 0.03])
            self.sliders['slit_width'] = Slider(
                ax, 'Ancho rendija (mm)', 0.01, 0.5, 
                valinit=self.params['slit_width']*1e3, valstep=0.01
            )
            self.sliders['slit_width'].on_changed(self.actualizar)
            
            # Separación
            ax = plt.axes([0.15, y_pos-0.05, 0.3, 0.03])
            self.sliders['slit_separation'] = Slider(
                ax, 'Separación (mm)', 0.1, 2.0, 
                valinit=self.params['slit_separation']*1e3, valstep=0.05
            )
            self.sliders['slit_separation'].on_changed(self.actualizar)
            
        elif self.apertura_tipo == 'Rejilla':
            # Diámetro de círculos
            ax = plt.axes([0.15, y_pos, 0.3, 0.03])
            self.sliders['grid_diameter'] = Slider(
                ax, 'Diámetro (mm)', 0.01, 0.5, 
                valinit=self.params['grid_diameter']*1e3, valstep=0.01
            )
            self.sliders['grid_diameter'].on_changed(self.actualizar)
            
            # Espaciado
            ax = plt.axes([0.15, y_pos-0.05, 0.3, 0.03])
            self.sliders['grid_spacing'] = Slider(
                ax, 'Espaciado (mm)', 0.1, 1.0, 
                valinit=self.params['grid_spacing']*1e3, valstep=0.05
            )
            self.sliders['grid_spacing'].on_changed(self.actualizar)
            
            # Desorden
            ax = plt.axes([0.15, y_pos-0.10, 0.3, 0.03])
            self.sliders['grid_disorder'] = Slider(
                ax, 'Desorden', 0.0, 0.5, 
                valinit=self.params['grid_disorder'], valstep=0.05
            )
            self.sliders['grid_disorder'].on_changed(self.actualizar)
    
    def cambiar_abertura(self, label):
        """Cambia el tipo de abertura y actualiza controles."""
        self.apertura_tipo = label
        self.crear_controles_abertura()
        self.actualizar(None)
        plt.draw()
    
    def actualizar(self, val):
        """Actualiza toda la visualización cuando cambian los parámetros."""
        # Actualizar parámetros desde sliders
        self.wavelength = self.sliders['wavelength'].val * 1e-9
        self.z = self.sliders['distance'].val
        self.zoom = self.sliders['zoom'].val
        
        # Actualizar parámetros específicos de abertura
        if 'circle_diameter' in self.sliders:
            self.params['circle_diameter'] = self.sliders['circle_diameter'].val * 1e-3
        if 'circle_eccentricity' in self.sliders:
            self.params['circle_eccentricity'] = self.sliders['circle_eccentricity'].val
        if 'rect_width' in self.sliders:
            self.params['rect_width'] = self.sliders['rect_width'].val * 1e-3
        if 'rect_height' in self.sliders:
            self.params['rect_height'] = self.sliders['rect_height'].val * 1e-3
        if 'slit_width' in self.sliders:
            self.params['slit_width'] = self.sliders['slit_width'].val * 1e-3
        if 'slit_separation' in self.sliders:
            self.params['slit_separation'] = self.sliders['slit_separation'].val * 1e-3
        if 'grid_diameter' in self.sliders:
            self.params['grid_diameter'] = self.sliders['grid_diameter'].val * 1e-3
        if 'grid_spacing' in self.sliders:
            self.params['grid_spacing'] = self.sliders['grid_spacing'].val * 1e-3
        if 'grid_disorder' in self.sliders:
            self.params['grid_disorder'] = self.sliders['grid_disorder'].val
        
        # Actualizar color del láser
        color = self.wavelength_to_color(self.wavelength*1e9)
        if self.laser_beam:
            self.laser_beam.remove()
        
        # Dibujar haz con el color correcto
        beam_x = [0.9, 1.5, 4.5, 5.2]
        beam_y_top = [0.5, 0.7, 0.7, 0.9]
        beam_y_bottom = [0.5, 0.3, 0.3, 0.1]
        
        self.laser_beam = self.axes['laser'].fill_between(
            [0.9, 2.0], [0.5, 0.5], [0.5, 0.5], 
            color=color, alpha=0.3
        )
        
        # Crear máscara de abertura
        mascara = self.crear_mascara_abertura()
        
        # Visualizar abertura
        self.axes['apertura'].clear()
        self.axes['apertura'].set_title('Apertura', fontsize=14, fontweight='bold')
        extent = np.array([-self.N//2, self.N//2, -self.N//2, self.N//2]) * self.dx * 1e3
        self.axes['apertura'].imshow(mascara, extent=extent, cmap='gray_r', origin='lower')
        self.axes['apertura'].set_xlim(-3, 3)
        self.axes['apertura'].set_ylim(-3, 3)
        self.axes['apertura'].set_xlabel('x (mm)')
        self.axes['apertura'].set_ylabel('y (mm)')
        self.axes['apertura'].grid(True, alpha=0.3)
        
        # Calcular difracción
        intensidad, x_obs, y_obs = self.calcular_difraccion(mascara)
        
        # Visualizar patrón de difracción
        self.axes['difraccion'].clear()
        self.axes['difraccion'].set_title('Patrón de Difracción', fontsize=14, fontweight='bold')
        
        # Aplicar zoom
        zoom_m = self.zoom * 1e-3
        extent = [-zoom_m*1e3, zoom_m*1e3, -zoom_m*1e3, zoom_m*1e3]
        
        # Encontrar índices para el zoom
        idx_x = np.abs(x_obs) <= zoom_m
        idx_y = np.abs(y_obs) <= zoom_m
        idx_2d = np.outer(idx_y, idx_x)
        
        if np.any(idx_2d):
            intensidad_zoom = intensidad[idx_2d].reshape(np.sum(idx_y), np.sum(idx_x))
            
            # Crear colormap personalizado basado en la longitud de onda
            if self.wavelength < 500e-9:
                cmap = 'Blues'
            elif self.wavelength < 600e-9:
                cmap = 'Greens'
            else:
                cmap = 'Reds'
            
            self.axes['difraccion'].imshow(
                intensidad_zoom, extent=extent,
                cmap=cmap, origin='lower',
                norm=PowerNorm(gamma=0.5),
                interpolation='bilinear'
            )
        
        self.axes['difraccion'].set_xlabel("x' (mm)")
        self.axes['difraccion'].set_ylabel("y' (mm)")
        self.axes['difraccion'].grid(True, alpha=0.3)
        
        # Actualizar pantalla con patrón
        self.axes['pantalla'].clear()
        self.axes['pantalla'].set_xlim(0, 1)
        self.axes['pantalla'].set_ylim(0, 1)
        self.axes['pantalla'].axis('off')
        
        # Dibujar pantalla con patrón
        screen = Rectangle((0.8, 0.1), 0.1, 0.8, facecolor='black', edgecolor='gray')
        self.axes['pantalla'].add_patch(screen)
        
        # Añadir representación simplificada del patrón en la pantalla
        if np.any(idx_2d) and intensidad_zoom.size > 0:
            pattern = Circle((0.85, 0.5), 0.03, 
                           facecolor=color, alpha=float(np.max(intensidad_zoom)))
            self.axes['pantalla'].add_patch(pattern)
        
        self.axes['pantalla'].text(0.5, 0.05, 'Pantalla', ha='center', fontsize=12)
        
        # Actualizar perfil
        self.axes['perfil'].clear()
        self.axes['perfil'].set_title('Perfil de Intensidad Central', fontsize=12)
        
        if np.any(idx_2d) and intensidad_zoom.size > 0:
            centro_y = intensidad_zoom.shape[0] // 2
            perfil = intensidad_zoom[centro_y, :]
            x_perfil = np.linspace(-self.zoom, self.zoom, len(perfil))
            
            self.axes['perfil'].plot(x_perfil, perfil, color=color, linewidth=2)
            self.axes['perfil'].fill_between(x_perfil, 0, perfil, alpha=0.3, color=color)
        
        self.axes['perfil'].set_xlabel("x' (mm)")
        self.axes['perfil'].set_ylabel('Intensidad')
        self.axes['perfil'].set_ylim(0, 1.1)
        self.axes['perfil'].grid(True, alpha=0.3)
        
        # Información adicional
        info_text = f'λ = {self.wavelength*1e9:.0f} nm, z = {self.z:.1f} m'
        self.fig.text(0.5, 0.02, info_text, ha='center', fontsize=10)
        
        plt.draw()
    
    def run(self):
        """Ejecuta el simulador interactivo."""
        self.crear_interfaz()
        plt.show()

# Función para ejecutar el simulador
def ejecutar_simulador_interactivo():
    """Ejecuta el simulador interactivo de difracción."""
    print("🔬 Iniciando Simulador Interactivo de Difracción de Fraunhofer")
    print("=" * 60)
    print("Instrucciones:")
    print("- Use los controles deslizantes para ajustar los parámetros")
    print("- Seleccione diferentes tipos de abertura con los botones")
    print("- El patrón se actualiza en tiempo real")
    print("=" * 60)
    
    simulador = SimuladorDifraccionInteractivo()
    simulador.run()

# Para Google Colab, usar ipywidgets si está disponible
try:
    import ipywidgets as widgets
    from IPython.display import display
    
    def crear_interfaz_colab():
        """Crea una interfaz específica para Google Colab usando ipywidgets."""
        simulador = SimuladorDifraccionInteractivo()
        
        # Crear widgets
        w_wavelength = widgets.IntSlider(
            value=550, min=380, max=780, step=10,
            description='λ (nm):', style={'description_width': 'initial'}
        )
        
        w_distance = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Distancia z (m):', style={'description_width': 'initial'}
        )
        
        w_zoom = widgets.IntSlider(
            value=10, min=1, max=50, step=1,
            description='Zoom (mm):', style={'description_width': 'initial'}
        )
        
        w_aperture = widgets.Dropdown(
            options=['Círculo', 'Rectángulo', 'Doble Rendija', 'Rejilla', 'Círculo + Cuadrado'],
            value='Círculo',
            description='Abertura:', style={'description_width': 'initial'}
        )
        
        # Widgets específicos por abertura
        w_diameter = widgets.FloatSlider(
            value=0.5, min=0.1, max=3.0, step=0.05,
            description='Diámetro (mm):', style={'description_width': 'initial'}
        )
        
        w_eccentricity = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.99, step=0.05,
            description='Excentricidad:', style={'description_width': 'initial'}
        )
        
        w_width = widgets.FloatSlider(
            value=0.5, min=0.1, max=3.0, step=0.05,
            description='Ancho (mm):', style={'description_width': 'initial'}
        )
        
        w_height = widgets.FloatSlider(
            value=1.0, min=0.1, max=3.0, step=0.05,
            description='Alto (mm):', style={'description_width': 'initial'}
        )
        
        w_slit_width = widgets.FloatSlider(
            value=0.05, min=0.01, max=0.5, step=0.01,
            description='Ancho rendija (mm):', style={'description_width': 'initial'}
        )
        
        w_slit_sep = widgets.FloatSlider(
            value=0.3, min=0.1, max=2.0, step=0.05,
            description='Separación (mm):', style={'description_width': 'initial'}
        )
        
        w_grid_diam = widgets.FloatSlider(
            value=0.1, min=0.01, max=0.5, step=0.01,
            description='Diám. círculos (mm):', style={'description_width': 'initial'}
        )
        
        w_grid_spacing = widgets.FloatSlider(
            value=0.4, min=0.1, max=1.0, step=0.05,
            description='Espaciado (mm):', style={'description_width': 'initial'}
        )
        
        w_disorder = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Desorden:', style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        # Contenedores para widgets específicos
        aperture_controls = widgets.VBox([])
        
        def update_aperture_controls(change):
            """Actualiza los controles según el tipo de abertura."""
            aperture_type = w_aperture.value
            
            if aperture_type == 'Círculo':
                aperture_controls.children = [w_diameter, w_eccentricity]
            elif aperture_type == 'Rectángulo':
                aperture_controls.children = [w_width, w_height]
            elif aperture_type == 'Doble Rendija':
                aperture_controls.children = [w_slit_width, w_slit_sep]
            elif aperture_type == 'Rejilla':
                aperture_controls.children = [w_grid_diam, w_grid_spacing, w_disorder]
            else:
                aperture_controls.children = [w_diameter, w_width]
        
        def update_simulation(change):
            """Actualiza la simulación con los nuevos parámetros."""
            with output:
                output.clear_output(wait=True)
                
                # Actualizar parámetros del simulador
                simulador.wavelength = w_wavelength.value * 1e-9
                simulador.z = w_distance.value
                simulador.zoom = w_zoom.value
                simulador.apertura_tipo = w_aperture.value
                
                # Actualizar parámetros específicos
                if simulador.apertura_tipo == 'Círculo':
                    simulador.params['circle_diameter'] = w_diameter.value * 1e-3
                    simulador.params['circle_eccentricity'] = w_eccentricity.value
                elif simulador.apertura_tipo == 'Rectángulo':
                    simulador.params['rect_width'] = w_width.value * 1e-3
                    simulador.params['rect_height'] = w_height.value * 1e-3
                elif simulador.apertura_tipo == 'Doble Rendija':
                    simulador.params['slit_width'] = w_slit_width.value * 1e-3
                    simulador.params['slit_separation'] = w_slit_sep.value * 1e-3
                elif simulador.apertura_tipo == 'Rejilla':
                    simulador.params['grid_diameter'] = w_grid_diam.value * 1e-3
                    simulador.params['grid_spacing'] = w_grid_spacing.value * 1e-3
                    simulador.params['grid_disorder'] = w_disorder.value
                
                # Mostrar simulación
                simulador.run_simulation(
                    simulador.apertura_tipo,
                    wavelength=simulador.wavelength,
                    z=simulador.z,
                    zoom=simulador.zoom,
                    colormap='inferno'
                )
                
                plt.show()
        
        # Conectar eventos
        w_aperture.observe(update_aperture_controls, names='value')
        w_aperture.observe(update_simulation, names='value')
        
        # Conectar todos los widgets a la actualización
        for widget in [w_wavelength, w_distance, w_zoom, w_diameter, w_eccentricity,
                      w_width, w_height, w_slit_width, w_slit_sep, 
                      w_grid_diam, w_grid_spacing, w_disorder]:
            widget.observe(update_simulation, names='value')
        
        # Interfaz principal
        main_controls = widgets.VBox([
            widgets.HTML('<h3>Simulador Interactivo de Difracción</h3>'),
            w_wavelength,
            w_distance,
            w_zoom,
            widgets.HTML('<hr>'),
            w_aperture,
            aperture_controls,
            widgets.HTML('<hr>')
        ])
        
        # Inicializar
        update_aperture_controls(None)
        update_simulation(None)
        
        # Mostrar interfaz
        display(widgets.HBox([main_controls, output]))
    
    # Si estamos en Colab, usar la interfaz de ipywidgets
    print("\n💡 Para una mejor experiencia interactiva en Google Colab, ejecute:")
    print("crear_interfaz_colab()")
    
except ImportError:
    print("\n⚠️ ipywidgets no está instalado. Use la interfaz estándar con:")
    print("ejecutar_simulador_interactivo()")

# Ejecutar automáticamente si es el script principal
if __name__ == "__main__":
    # Intentar usar la versión de Colab primero
    try:
        crear_interfaz_colab()
    except:
        # Si falla, usar la versión estándar
        ejecutar_simulador_interactivo()