"""



"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.fft import fft2, fftshift, fftfreq

from matplotlib.colors import PowerNorm

from matplotlib.patches import Circle, Rectangle



# Configuración global para mejor calidad

plt.rcParams["figure.dpi"] = 120

plt.rcParams['figure.facecolor'] = 'black'

plt.rcParams['text.color'] = 'white'

plt.rcParams['axes.labelcolor'] = 'white'

plt.rcParams['xtick.color'] = 'white'

plt.rcParams['ytick.color'] = 'white'

plt.rcParams['axes.edgecolor'] = 'white'

plt.rcParams['axes.titlecolor'] = 'white'



class SimuladorDifraccion2D:

    """

    Simulador interactivo de difracción de Fraunhofer con cálculos físicamente correctos.

    """



    def __init__(self):

        """Inicializa el simulador con parámetros por defecto."""

        # Parámetros físicos por defecto

        self.parametros = {

            'N': 512,               # Tamaño de la matriz

            'dx': 5e-6,            # Tamaño del pixel en metros (5 μm)

            'z': 1.0,              # Distancia de propagación en metros

            'wavelength': 550e-9,   # Longitud de onda en metros

            'zoom': 10.0,          # Zoom en mm para la visualización

        }



        # Parámetros específicos de cada abertura

        self.params_abertura = {

            'circle_diameter': 1.0e-3,      # 1 mm

            'rect_width': 0.5e-3,           # 0.5 mm

            'rect_height': 2.0e-3,          # 2 mm

            'slit_width': 0.05e-3,          # 0.05 mm

            'slit_separation': 0.3e-3,      # 0.3 mm

            'slit_height': 2.0e-3,          # 2 mm

            'grid_diameter': 0.2e-3,        # 0.2 mm

            'grid_spacing': 0.8e-3,         # 0.8 mm

        }



        self.escena_actual = 'Círculo'

        self.escala_log = False

        self.colormap = 'viridis'



    def _crear_mascara_abertura(self):

        """Crea la máscara de la abertura según la escena seleccionada."""

        N = self.parametros['N']

        dx = self.parametros['dx']



        # Crear grilla de coordenadas

        x = np.arange(-N//2, N//2) * dx

        y = np.arange(-N//2, N//2) * dx

        X, Y = np.meshgrid(x, y)



        mascara = np.zeros((N, N))



        if self.escena_actual == 'Círculo':

            radio = self.params_abertura['circle_diameter'] / 2

            mascara = (X**2 + Y**2 <= radio**2).astype(float)



        elif self.escena_actual == 'Rectángulo':

            ancho = self.params_abertura['rect_width']

            alto = self.params_abertura['rect_height']

            mascara = (np.abs(X) <= ancho/2) & (np.abs(Y) <= alto/2)

            mascara = mascara.astype(float)



        elif self.escena_actual == 'Doble Rendija':

            ancho = self.params_abertura['slit_width']

            sep = self.params_abertura['slit_separation']

            alto = self.params_abertura['slit_height']



            mascara1 = (np.abs(X - sep/2) <= ancho/2) & (np.abs(Y) <= alto/2)

            mascara2 = (np.abs(X + sep/2) <= ancho/2) & (np.abs(Y) <= alto/2)

            mascara = (mascara1 | mascara2).astype(float)



        elif self.escena_actual == 'Rejilla de Círculos':

            radio = self.params_abertura['grid_diameter'] / 2

            espaciado = self.params_abertura['grid_spacing']



            for i in range(-2, 3):

                for j in range(-2, 3):

                    x_centro = i * espaciado

                    y_centro = j * espaciado

                    mascara_temp = ((X - x_centro)**2 + (Y - y_centro)**2 <= radio**2)

                    mascara = np.maximum(mascara, mascara_temp)



        elif self.escena_actual == 'Círculo + Cuadrado':

            # Círculo

            radio = self.params_abertura['circle_diameter'] / 2

            mascara_circulo = (X**2 + Y**2 <= radio**2)



            # Cuadrado

            lado = self.params_abertura['rect_width']

            mascara_cuadrado = (np.abs(X) <= lado/2) & (np.abs(Y) <= lado/2)



            mascara = np.maximum(mascara_circulo, mascara_cuadrado).astype(float)



        return mascara



    def _calcular_difraccion_fraunhofer(self, mascara):

        """Calcula el patrón de difracción usando FFT."""

        dx = self.parametros['dx']

        wavelength = self.parametros['wavelength']

        z = self.parametros['z']

        N = self.parametros['N']



        # FFT de la abertura

        campo_fft = fftshift(fft2(mascara)) * dx**2



        # Frecuencias espaciales

        fx = fftshift(fftfreq(N, dx))

        fy = fftshift(fftfreq(N, dx))



        # Coordenadas en el plano de observación

        x_obs = wavelength * z * fx

        y_obs = wavelength * z * fy



        # Intensidad

        intensidad = np.abs(campo_fft)**2



        # Normalizar

        intensidad = intensidad / np.max(intensidad)



        return intensidad, x_obs, y_obs



    def _wavelength_to_color(self, wavelength_nm):

        """Convierte longitud de onda a color RGB."""

        if wavelength_nm < 440:

            return (0.5, 0.0, 1.0)

        elif wavelength_nm < 490:

            t = (wavelength_nm - 440) / 50

            return (0.5*(1-t), 0, 1.0)

        elif wavelength_nm < 510:

            t = (wavelength_nm - 490) / 20

            return (0, t, 1.0)

        elif wavelength_nm < 580:

            t = (wavelength_nm - 510) / 70

            return (0, 1.0, 1-t)

        elif wavelength_nm < 645:

            t = (wavelength_nm - 580) / 65

            return (t, 1.0, 0)

        else:

            t = min(1.0, (wavelength_nm - 645) / 55)

            return (1.0, 1-t, 0)



    def run_simulation(self, escena, wavelength=550e-9, z=1.0, zoom=10.0, escala_log=False, colormap='viridis'):

        """Corre la simulación y muestra los resultados."""

        self.escena_actual = escena

        self.parametros['wavelength'] = wavelength

        self.parametros['z'] = z

        self.parametros['zoom'] = zoom

        self.escala_log = escala_log

        self.colormap = colormap



        try:

            # Crear figura con estilo mejorado

            fig = plt.figure(figsize=(18, 6))



            # 1. Abertura

            ax1 = plt.subplot(131)

            mascara = self._crear_mascara_abertura()



            # Visualización de la abertura

            extent_abertura = np.array([-self.parametros['N']//2, self.parametros['N']//2,

                                       -self.parametros['N']//2, self.parametros['N']//2]) * self.parametros['dx'] * 1e3



            im1 = ax1.imshow(mascara, extent=extent_abertura, cmap='gray_r', origin='lower')

            ax1.set_title(f'Abertura: {self.escena_actual}', fontsize=14, fontweight='bold', pad=15)

            ax1.set_xlabel('x (mm)', fontsize=12)

            ax1.set_ylabel('y (mm)', fontsize=12)

            ax1.grid(True, alpha=0.3, linestyle='--')



            # Ajustar límites para mejor visualización

            max_size = 3.0  # mm

            ax1.set_xlim(-max_size, max_size)

            ax1.set_ylim(-max_size, max_size)



            # 2. Patrón de difracción

            ax2 = plt.subplot(132)

            intensidad, x_obs, y_obs = self._calcular_difraccion_fraunhofer(mascara)



            # Aplicar zoom

            zoom_m = self.parametros['zoom'] * 1e-3  # convertir mm a m

            extent_difraccion = [-zoom_m, zoom_m, -zoom_m, zoom_m]



            # Encontrar índices para el zoom

            idx_x = np.abs(x_obs) <= zoom_m

            idx_y = np.abs(y_obs) <= zoom_m

            idx_2d = np.outer(idx_y, idx_x)



            # Extraer región con zoom

            intensidad_zoom = intensidad[idx_2d].reshape(np.sum(idx_y), np.sum(idx_x))



            if self.escala_log:

                datos_mostrar = np.log10(intensidad_zoom + 1e-10)

                label_cbar = 'log₁₀(Intensidad)'

                norm = None

            else:

                datos_mostrar = intensidad_zoom

                label_cbar = 'Intensidad'

                norm = PowerNorm(gamma=0.5)  # Realce de contraste



            im2 = ax2.imshow(datos_mostrar, extent=[e*1e3 for e in extent_difraccion],

                           cmap=self.colormap, origin='lower',

                           norm=norm, interpolation='bilinear')



            ax2.set_title(f'Patrón de Difracción (λ = {self.parametros["wavelength"]*1e9:.0f} nm, z = {self.parametros["z"]:.2f} m)',

                        fontsize=14, fontweight='bold', pad=15)

            ax2.set_xlabel("x' (mm)", fontsize=12)

            ax2.set_ylabel("y' (mm)", fontsize=12)

            ax2.grid(True, alpha=0.3, linestyle='--')



            # Colorbar

            cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            cbar.set_label(label_cbar, rotation=270, labelpad=20, fontsize=11)



            # 3. Perfil central

            ax3 = plt.subplot(133)



            # Extraer perfil central

            centro_y = intensidad_zoom.shape[0] // 2

            perfil = intensidad_zoom[centro_y, :]

            x_perfil = np.linspace(-self.parametros['zoom'], self.parametros['zoom'], len(perfil))



            if self.escala_log:

                perfil_plot = np.log10(perfil + 1e-10)

                ylabel = 'log₁₀(Intensidad)'

            else:

                perfil_plot = perfil

                ylabel = 'Intensidad'



            # Color basado en longitud de onda

            color = self._wavelength_to_color(self.parametros['wavelength'] * 1e9)



            ax3.plot(x_perfil, perfil_plot, color=color, linewidth=2.5, label=f'λ = {self.parametros["wavelength"]*1e9:.0f} nm')

            ax3.fill_between(x_perfil, np.min(perfil_plot), perfil_plot, alpha=0.3, color=color)



            ax3.set_title('Perfil de Intensidad Central', fontsize=14, fontweight='bold', pad=15)

            ax3.set_xlabel("x' (mm)", fontsize=12)

            ax3.set_ylabel(ylabel, fontsize=12)

            ax3.grid(True, alpha=0.3, linestyle='--')

            ax3.legend(fontsize=10)



            # Verificar criterio de campo lejano

            tamano_max = np.max([

                self.params_abertura['circle_diameter'],

                self.params_abertura['rect_width'],

                self.params_abertura['rect_height'],

                5 * self.params_abertura['grid_spacing']

            ])



            criterio_fraunhofer = self.parametros['z'] > 10 * (tamano_max**2) / self.parametros['wavelength']



            if not criterio_fraunhofer:

                fig.text(0.5, 0.02,

                       '⚠️ Advertencia: La distancia z podría no cumplir el criterio de campo lejano (z >> D²/λ)',

                       ha='center', color='red', fontsize=11,

                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))



            plt.tight_layout()

            filename = f"punto 2/difraccion_{self.escena_actual}_wl_{self.parametros['wavelength']*1e9:.0f}_z_{self.parametros['z']:.2f}.png"

            plt.savefig(filename, facecolor='black')

            plt.close(fig)

            print(f"Imagen guardada en {filename}")



        except Exception as e:

            print(f"❌ Error: {e}")

            import traceback

            traceback.print_exc()



def ejecutar_simulador():

    """Ejecuta el simulador de difracción para una variedad de configuraciones."""

    print("🔬 Generando simulaciones de Difracción de Fraunhofer 2D")

    print("=" * 60)



    simulador = SimuladorDifraccion2D()

   

    escenas = ['Círculo', 'Rectángulo', 'Doble Rendija', 'Rejilla de Círculos']

    longitudes_onda = [450e-9, 550e-9, 650e-9] # Azul, Verde, Rojo



    for escena in escenas:

        for wl in longitudes_onda:

            print(f"Simulando: {escena} con λ = {wl*1e9:.0f} nm")

            simulador.run_simulation(escena, wavelength=wl, colormap='inferno')



    print("=" * 60)

    print("✅ Simulaciones completadas.")





if __name__ == "__main__":

    ejecutar_simulador()