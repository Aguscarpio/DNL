#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:34:23 2020

@author: aguscarpio
"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import *

x = symbols('x')
y = symbols('y')

#-------------SETTINGS------------------
xdot = y
ydot = -2*x-3*y

normalize = True
stream_plot = False

# Rango del espacio a graficar
minx, maxx = -3, 3
miny, maxy = -3, 3

# Cantidad de flechas en x y en y (el producto es el total)
xarrows, yarrows = 30, 30
#---------------------------------------

# Funcion principal
def two_D_flow(xdot,ydot,norm, streamplot):
    '''
    Graficar el flujo 2D dadas las expresiones de x_dot e y_dot
    norm: True para normalizar los vectores, False si no
    streamplot: True para hacer streamplot en lugar de quiver
    '''

    # Conviene tenerlo como función para flexibilidad del codigo
    def eval(x, y, expr):
        return expr(x,y)

    # Lambdifico las expresiones de dxdt y dydt
    x_dot = lambdify([x,y], xdot)
    y_dot = lambdify([x,y], ydot)

    # Defino el espacio y armo la grilla
    xs = np.linspace(minx, maxx, xarrows)
    ys = np.linspace(miny, maxy, yarrows)
    xx,yy = np.meshgrid(xs,ys, sparse=True)

    # Calculo el valor del campo para toda la grilla
    dy = eval(xx,yy, y_dot)
    dx = eval(xx,yy, x_dot)

    # Por default no normalizar (factor de normalizacion = 1)
    norm_term = np.ones(shape=(yarrows,xarrows))

    # Defino un vector de colores en función del modulo de cada vector
    colors = np.hypot(dx, dy)
    # Actualizo el normalizador en caso de querer hacerlo
    if norm:
        norm_term = np.sqrt(dx**2 + dy**2)
    # Streamploteo en caso de quere hacerlo
    if streamplot:
        fig = plt.streamplot(xx,yy,
                dx/norm_term, dy/norm_term)
        plt.show()
        return

    # Declaro la figura
    fig = plt.quiver(xx,yy,dx/norm_term,
            dy/norm_term, colors,
            cmap='RdYlGn_r', pivot = 'mid')
    # Barra de color
    cb = plt.colorbar(fig)
    plt.show()

two_D_flow(xdot,ydot,normalize,stream_plot)
