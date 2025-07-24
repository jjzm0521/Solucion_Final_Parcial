import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.special import fresnel

# ==============================================================================
# Contenido de: core/physics.py
# ==============================================================================

PI = np.pi

def _to_scipy_arg(u):
    """Convierte la variable física al argumento para las funciones Fresnel de SciPy."""
    return np.sqrt(2 / PI) * u

def fresnel_CS_from_eta(eta):
    """
    Calcula las integrales de Fresnel C y S a partir de la variable física eta.

    Args:
        eta (np.ndarray): La variable física.

    Returns:
        tuple: Una tupla que contiene las integrales C y S de Fresnel.
    """
    U = _to_scipy_arg(eta)
    S, C = fresnel(U)
    return C, S

# ==============================================================================
# Contenido de: core/propagation.py
# ==============================================================================

def field_slit(x, a, lam, z):
    """Calcula el campo relativo para una rendija única."""
    scale = np.sqrt(2 / (lam * z))
    u1 = scale * (x - a)
    u2 = scale * (x + a)
    C1, S1 = fresnel_CS_from_eta(u1)
    C2, S2 = fresnel_CS_from_eta(u2)
    return (C2 - C1) + 1j * (S2 - S1)

def field_double_slit(x, a, d, lam, z):
    """Calcula el campo relativo para una doble rendija."""
    return field_slit(x - d / 2, a, lam, z) + field_slit(x + d / 2, a, lam, z)

def field_half_plane(x, lam, z, side='right'):
    """Calcula el campo relativo para un semiplano."""
    scale = np.sqrt(2 / (lam * z))
    eta = scale * x if side == 'right' else -scale * x
    C, S = fresnel_CS_from_eta(eta)
    return (0.5 + C) + 1j * (0.5 + S)

def intensity(U):
    """Calcula la intensidad (magnitud al cuadrado) a partir del campo complejo."""
    return np.abs(U)**2

# ==============================================================================
# Simulador Interactivo con Matplotlib Widgets
# ==============================================================================

# --- Configuración de la figura y los ejes ---
fig = plt.figure(figsize=(10, 9))
fig.subplots_adjust(left=0.1, bottom=0.4, hspace=0.5)

ax_2d = fig.add_subplot(2, 1, 1)
ax_1d = fig.add_subplot(2, 1, 2)

# --- Parámetros iniciales ---
init_geom = 'Rendija simple'
init_lam_nm = 633.0
init_a_um = 50.0
init_d_um = 200.0
init_z_mm = 500.0
init_x_range_a = 10.0
init_n_points = 1000

# --- Función para calcular los datos de difracción ---
def get_diffraction_data(geometry, lam_nm, a_um, d_um, z_mm, x_range_a, n_points):
    lam = lam_nm * 1e-9
    a = a_um * 1e-6
    d = d_um * 1e-6
    z = z_mm * 1e-3
    # Para el semiplano, el rango no debe depender de 'a'
    x_max = x_range_a * a if geometry != 'Semiplano' else x_range_a * 1e-4
    x = np.linspace(-x_max, x_max, int(n_points))

    title = "Patrón de Difracción"
    if geometry == 'Rendija simple':
        U = field_slit(x, a, lam, z)
        fresnel_number = a**2 / (lam * z)
        title = f'Perfil de Intensidad - Rendija Simple (F = {fresnel_number:.2f})'
    elif geometry == 'Doble rendija':
        U = field_double_slit(x, a, d, lam, z)
        fresnel_number = a**2 / (lam * z)
        title = f'Perfil de Intensidad - Doble Rendija (F = {fresnel_number:.2f})'
    elif geometry == 'Semiplano':
        U = field_half_plane(x, lam, z)
        fresnel_number = x_max**2 / (lam * z)
        title = f'Perfil de Intensidad - Semiplano (F_ref = {fresnel_number:.2f})'
    else:
        return np.linspace(-1, 1, int(n_points)), np.zeros(int(n_points)), "Seleccione una geometría"

    I = intensity(U)

    # Lógica de normalización
    if geometry == 'Semiplano':
        # Normalizar a la intensidad asintótica (sin obstrucción), que es 2.0
        I /= 2.0
    else:
        # Para rendijas, normalizar al valor máximo
        if I.max() > 0:
            I /= I.max()

    return x, I, title

# --- Graficar los datos iniciales ---
x_init, I_init, title_init = get_diffraction_data(init_geom, init_lam_nm, init_a_um, init_d_um, init_z_mm, init_x_range_a, init_n_points)
pattern_2d_init = np.tile(I_init, (100, 1))

# CAMBIO: Se cambió cmap='gray_r' a cmap='gray' para invertir los colores.
im = ax_2d.imshow(pattern_2d_init, cmap='gray', aspect='auto', extent=[x_init.min() * 1e3, x_init.max() * 1e3, 0, 1])
ax_2d.set_title("Visualización del Patrón de Difracción")
ax_2d.set_yticks([])
ax_2d.set_xlabel('Posición en la pantalla (mm)')

line, = ax_1d.plot(x_init * 1e3, I_init, color='dodgerblue')
ax_1d.set_xlabel('Posición en la pantalla (mm)')
ax_1d.set_ylabel('Intensidad Normalizada')
ax_1d.set_title(title_init)
ax_1d.grid(True)
ax_1d.set_ylim(0, 1.5) # Aumentar límite para ver el sobreimpulso

# --- Definir la posición de los widgets ---
axcolor = 'lightgoldenrodyellow'
ax_geom = plt.axes([0.1, 0.25, 0.18, 0.1], facecolor=axcolor)
ax_lam = plt.axes([0.35, 0.21, 0.55, 0.02], facecolor=axcolor)
ax_a = plt.axes([0.35, 0.18, 0.55, 0.02], facecolor=axcolor)
ax_d = plt.axes([0.35, 0.15, 0.55, 0.02], facecolor=axcolor)
ax_z = plt.axes([0.35, 0.12, 0.55, 0.02], facecolor=axcolor)
ax_x_range = plt.axes([0.35, 0.09, 0.55, 0.02], facecolor=axcolor)
ax_n_points = plt.axes([0.35, 0.06, 0.55, 0.02], facecolor=axcolor)

# --- Crear los widgets ---
radio_geom = RadioButtons(ax_geom, ('Rendija simple', 'Doble rendija', 'Semiplano'), active=0)
slider_lam = Slider(ax_lam, 'λ (nm)', 300, 1000, valinit=init_lam_nm)
slider_a = Slider(ax_a, 'a (µm)', 5, 1000, valinit=init_a_um)
slider_d = Slider(ax_d, 'd (µm)', 10, 2000, valinit=init_d_um)
slider_z = Slider(ax_z, 'z (mm)', 1, 5000, valinit=init_z_mm)
slider_x_range = Slider(ax_x_range, 'Rango X', 1, 20, valinit=init_x_range_a)
slider_n_points = Slider(ax_n_points, 'N Puntos', 100, 5000, valinit=init_n_points, valstep=100)

# --- Función para actualizar los gráficos ---
def update(val):
    geom = radio_geom.value_selected
    lam_nm = slider_lam.val
    a_um = slider_a.val
    d_um = slider_d.val
    z_mm = slider_z.val
    x_range_a = slider_x_range.val
    n_points = slider_n_points.val

    slider_d.ax.set_visible(geom == 'Doble rendija')
    slider_a.ax.set_visible(geom != 'Semiplano')
    slider_x_range.label.set_text("Rango X/a" if geom != 'Semiplano' else "Rango X (x1e-4 m)")


    x, I, title = get_diffraction_data(geom, lam_nm, a_um, d_um, z_mm, x_range_a, n_points)
    
    pattern_2d = np.tile(I, (100, 1))
    
    im.set_data(pattern_2d)
    im.set_extent([x.min() * 1e3, x.max() * 1e3, 0, 1])
    im.set_clim(vmin=0, vmax=I.max())
    ax_2d.set_xlim(x.min() * 1e3, x.max() * 1e3)
    
    line.set_data(x * 1e3, I)
    ax_1d.set_title(title)
    
    ax_1d.relim()
    ax_1d.autoscale_view()
    ax_1d.set_ylim(0, 1.5)
    
    fig.canvas.draw_idle()

# --- Conectar los widgets a la función de actualización ---
radio_geom.on_clicked(update)
slider_lam.on_changed(update)
slider_a.on_changed(update)
slider_d.on_changed(update)
slider_z.on_changed(update)
slider_x_range.on_changed(update)
slider_n_points.on_changed(update)

update(None)

plt.show()
# --- Fin del código ---
# Este código crea una simulación interactiva del patrón de difracción para diferentes geometrías