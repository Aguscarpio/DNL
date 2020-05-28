import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sympy import *

# Variables
x = symbols('x')
y = symbols('y')

# Parametros
a = symbols('a')
b = symbols('b')

#-----------SETTINGS------------

# Expresiones de xdot e ydot
xdot = 1+x**2-y
ydot = a*(b*x-y)

# Rango del espacio a graficar
minx, maxx = -5, 5
miny, maxy = -5, 5

# Rango de los parámetros
mina, maxa = -4, 4
minb, maxb = -4, 4

# Cantidad de flechas en x y en y (el producto es el total)
xarrows, yarrows = 30, 30

#-------------------------------

# Definimos la grilla del espacio a graficar
X, Y = np.mgrid[minx:maxx:(maxx-minx)/xarrows,
                miny:maxy:(maxy-miny)/yarrows]

# Funcion conveniente para que no queden choclazos
def eval(x,y,a,b,expr):
    return expr(x,y,a,b)

# Lambdificamos
x_dot = lambdify([x,y,a,b], xdot)
y_dot = lambdify([x,y,a,b], ydot)

# Valores iniciales de a y b
a0, b0 = 0, 0

# Evaluamos el campo
dx = eval(X, Y, a0, b0, x_dot)
dy = eval(X, Y, a0, b0, y_dot)

fig, ax = plt.subplots()

# Termino de normalizacion y escala de color proporcional al campo
norm_term = np.sqrt(dx**2+dy**2)
colors = np.hypot(dx, dy)

# Importante asignar el quiver a una variable (Q) ademas de plotearlo
Q = plt.quiver(X, Y, dx/norm_term, dy/norm_term,
                colors, cmap = 'RdYlGn_r', pivot = 'mid')

# Truquito para que no se superponga el plot con los sliders
plt.subplots_adjust(bottom=0.25)
plt.axis([minx, maxx, miny, maxy])

# Posicion y tamaño donde voy a querer los sliders
axa = plt.axes([0.15, 0.10, 0.75, 0.03])
axb = plt.axes([0.15, 0.05, 0.75, 0.03])

# Defino los sliders. Le doy posicion en el plot, rango y valor inicial
sa = Slider(axa, 'a', mina, maxa, valinit=0)
sb = Slider(axb, 'b', minb, maxb, valinit=0)

# Hasta acá no fue necesario un enfoque funcional.
# Ahora que necesitamos actualizar el plot, si es necesario.

def update(val):
    # Evaluamos el campo en cada x e y con el valor actualizado de a y b
    # Notar que sa es un objeto de tipo slider y val es un atributo (su valor)
    dx = eval(X, Y, sa.val, sb.val, x_dot)
    dy = eval(X, Y, sa.val, sb.val, y_dot)

    norm_term = np.sqrt(dx**2+dy**2)

    # Cambiamos los vectores de nuestro quiver Q definido anteriormente
    # Esto esta buenisimo porque no tiene que generar la figura entera,
    # solo actualiza los direcciones de cada flecha. Por eso funca rapido.
    Q.set_UVC(dx/norm_term, dy/norm_term)

    fig.canvas.draw_idle()

# Si algun slider sufre algun cambio actualizamos
sa.on_changed(update)
sb.on_changed(update)

# Ya que estamos escribimos las expresiones de xdot e ydot en latex
plt.suptitle(f"$\dot{x} = {latex(xdot)}$" + "\n"
        + f"$\dot{y} = {latex(ydot)}$", fontsize=12)
plt.show()

# Comentario final:
# El codigo funciona si las ecuaciones dependen de ningun parametro
# o incluso si no dependen de ninguno. Sin embargo los sliders van a
# aparecer igual. Estaria bueno mejorar el codigo de modo tal que uno
# pueda definir al principio cuantos parametros tiene su problema
# (e incluso poder elegirle otros nombres que a y b) para que el
# resultado sea mas elegante sin necesidad de harcodear la funcion principal
