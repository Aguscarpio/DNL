# Hay algunas invocaciones de funciones que estan comentadas
# Para activarlas simplemente hay que borrar el '#'

# BUG para corregir:
# No encuentra nulclinas completamente verticales
# del tipo x = ...

import matplotlib.pyplot as plt
import numpy as np
from sympy import *

x = symbols('x')
y = symbols('y')

#----------SETTINGS--------
# valores de los parametros (jueguen)
a, b = 0.9, 2

# propongan el sistema de ecuaciones que quieran
xdot = x-y-a*x*y
ydot = x**2-y*b

minx, maxx = -4, 4
miny, maxy = -4, 4

#--------------------------
step = 1000

xs = np.linspace(minx,maxx,step)
ys = np.linspace(miny,maxy,step)

fig, ax = plt.subplots()

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14, rotation=0)

fp_pallete = {
        'saddle': '#3E0',
        'node': '#000',
        'spiral': '#B34',
        'center': '#0BB',
        'nifp': '#B7B',
        'degenerated or stars': '#FA6'
        }

def nulclines_plot(xdot, ydot, colors=["#F00","#00F"]):
    labels = [f'Nulclina de $\dot{x}$', f'Nulclina de $\dot{y}$']
    # Nulclinas de x e y
    xncs = solve(xdot, y)
    yncs = solve(ydot, y)
    # empaquetadas en un vector
    ncs = [xncs, yncs]
    # Para cada nulclina (diferenciando si es de x o de y)
    for i in range(len(ncs)):
        for c in ncs[i]:
            # Buscamos discontinuidades
            holes = list(singularities(c,x, Reals))+[maxx]
            hole = holes.pop(0)
            xplot = []
            yplot = []
            cl = lambdify(x,c)
            # Y las graficamos (evitando discontinuidades)
            for xx in xs:
                if xx>hole:
                    ax.plot(xplot,yplot,color=colors[i])
                    xplot=[]
                    yplot=[]
                    hole = holes.pop(0)
                yy = cl(xx)
                xplot.append(xx)
                yplot.append(yy)
            ax.plot(xplot, yplot,color=colors[i], label=labels[i])

def fixed_points(xdot, ydot):
    fps = set() #comienza siendo un conjunto vacio
    # Busco las intersecciones de las nulclinas
    for nx in solve(xdot, y):
        for ny in solve(ydot, y):
            for sx in solveset(nx-ny,x,domain=S.Reals):
                fps.add((sx, lambdify(x,ny)(sx)))
    return fps

# Calculo la matriz jacobiana
def jacobian_matrix(xdot, ydot):
    mdot = Matrix([xdot,ydot])
    mvar = Matrix([x,y])
    J = mdot.jacobian(mvar)
    return lambdify([x,y],J)

def fp_classifier(fixed_points):
    J = jacobian_matrix(xdot,ydot)
    fp_classified = []
    # Vamos a clasificar los puntos fijos
    for (x_star,y_star) in fixed_points:
        # Evaluamos el jacobiano en el punto fijo
        evaluated_jacobian = np.array(J(x_star, y_star), dtype=float)
        # Calculamos traza y determinante
        trace = np.trace(evaluated_jacobian)
        determinant = np.linalg.det(evaluated_jacobian)

        # En funcion de la relacion entre traza y determinante
        # clasificamos al punto fijo
        # Guardamos al punto fijo con su etiqueta en la lista fp_classified
        if determinant<0:
            fp_classified.append((x_star,y_star,"saddle"))
        elif determinant==0:
            fp_classified.append((x_star,y_star,"nifp"))
        else:
            if trace==0:
                fp_classified.append((x_star,y_star,"center"))
            elif trace**2<4*determinant:
                fp_classified.append((x_star,y_star,"spiral"))
            elif trace**2>4*determinant:
                fp_classified.append((x_star,y_star,"node"))
            else:
                fp_classified.append((x_star,y_star,"degenerated or stars"))
    # Devolvemos la misma lista de puntos fijos, ahora etiquetados
    return fp_classified

# Esta funcion grafica nulclinas y puntos fijos etiquetados
def plot_ncs_and_fps():
    # Usamos la funcion para clasificar pfs
    fps_tagged = fp_classifier(fixed_points(xdot,ydot))
    # Graficamos nulclinas
    nulclines_plot(xdot,ydot)
    # Y graficamos los puntos fijos con el label correspondiente
    for fp in fps_tagged:
        plt.plot(fp[0], fp[1], "o", markersize=8,
                color=fp_pallete[fp[2]], label=fp[2])
    # Añadimos leyenda
    plt.legend()



# Llegado este punto hay varias opciones para continuar

# Una opcion podria ser realizar un streamplot junto a
# las nulclinas y los puntos fijos

def stream_normal(xdot, ydot):
    x_dot = lambdify([x,y], xdot)
    y_dot = lambdify([x,y], ydot)
    xx,yy = np.meshgrid(xs, ys, sparse=True)
    # Suboptimo pero funciona
    dx = x_dot(xx,yy)/np.ones(shape=(step,step))
    dy = y_dot(xx,yy)/np.ones(shape=(step,step))

    ax.streamplot(xx, yy, dx, dy, color = "#000")


# O quizas preferis hacer un retrato de fases
# solo con curvas especialmente seleccionadas
# ---- EN CONSTRUCCION----
def stream_special(xdot, ydot):
    def initial_conditions(fp, J):
        # Estos parametros son re arbitrarios
        # si no obtienen los graficos deseados jueguen un poco
        epsilon = 1
        mm = 1
        keps = 0.5
        evted_jac = np.array(J(fp[0], fp[1]), dtype=float)
        if fp[2]=="saddle":
            eigvals, eigvecs = np.linalg.eig(evted_jac)
            for (l, v) in zip(eigvals, eigvecs):
                if l>0:
                    continue
                else:
                    # Elegimos condiciones iniciales en el entorno
                    # de los autovectores
                    x1, y1 = fp[0]+epsilon*v[0]+mm, fp[1]+epsilon*v[1]+mm
                    x2, y2 = fp[0]+epsilon*v[0]-mm, fp[1]+epsilon*v[1]-mm
                    x3, y3 = fp[0]-epsilon*v[0]+mm, fp[1]-epsilon*v[1]+mm
                    x4, y4 = fp[0]-epsilon*v[0]-mm, fp[1]-epsilon*v[1]-mm
                    return [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        if fp[2] in ["spiral"]:
            # Mas arbitrariedades (pero increiblemente funciona bien)
            return [(fp[0]+mm, fp[1]+0.01), (fp[0]+mm, fp[1]+0.21)]
        if fp[2] in ["node", "nifp", "degenerated or stars"]:
            eigvals, eigvecs = np.linalg.eig(evted_jac)
            initconds = []
            for (l, v) in zip(eigvals, eigvecs):
                initconds.append((fp[0]+keps*v[0], fp[1]+keps*v[1]))
                initconds.append((fp[0]-keps*v[0], fp[1]-keps*v[1]))
            return initconds
    fps_tagged = fp_classifier(fixed_points(xdot,ydot))
    x_dot = lambdify([x,y], xdot)
    y_dot = lambdify([x,y], ydot)
    xx,yy = np.meshgrid(xs, ys, sparse=True)
    # Suboptimo pero funciona
    dx = x_dot(xx,yy)/np.ones(shape=(step,step))
    dy = y_dot(xx,yy)/np.ones(shape=(step,step))

    for fp in fps_tagged:
        col = fp_pallete[fp[2]]
        #col="#000" #Descomentar si queres todas las curvas negras
        ax.streamplot(xx, yy, dx, dy, color=col,
        start_points=initial_conditions(fp, J=jacobian_matrix(xdot,ydot)),
        minlength=0.0001)

# INVOCACION DE FUNCIONES
# Comentar y descomentar a gusto

plot_ncs_and_fps()
stream_normal(xdot, ydot)
stream_special(xdot, ydot)

#----------------------------------

plt.suptitle(f"$\dot{x} = {latex(xdot)}$" + "\n"
        + f"$\dot{y} = {latex(ydot)}$", fontsize=12)
plt.show()





