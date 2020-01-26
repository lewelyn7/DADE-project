import numpy as np 
import matplotlib.pyplot as plt
from functools import partial



def cheops(x_top, y_top, x, y):
    leng = 1
    if abs(x-x_top) < leng and abs(y-y_top) < leng:
        return (leng-abs(x-x_top))*(leng-abs(y-y_top))
    else:
        return 0




def integral(xy_function, start, end, var, dwhat):
    step = 0.01
    sum = 0
    if dwhat == "dx":
        for i in np.arange(start, end, step):
            sum += xy_function(i+ step/2, var) * step
    if dwhat == "dy":
        for i in np.arange(start, end, step):
            sum += xy_function(var, i+ step/2) * step
    return sum


def double_integral(xy_function, x_start, x_end, y_start, y_end):

    sum = 0
    step = 0.01
    base_area = step*step
    for x in np.arange(x_start, x_end, step):
        for y in np.arange(y_start, y_end, step):
            x2 = x + step/2
            y2 = y + step/2
            sum += base_area * xy_function(x2,y2)
    return sum

def double_derivative(xy_function, var, x ,y):
    h = 0.01
    if var == "x":
        return (xy_function(x + h, y) - xy_function(x - h, y))/(2*h)
    elif var == "y":
        return (xy_function(x, y + h) - xy_function(x, y-h))/(2*h)
    else:
        raise Exception("not x nor y exception")

def base_func_generator(top_xy_vertices):
    base_functions = []
    for xy_vertex in top_xy_vertices:
        x , y = xy_vertex
        base_functions.append(partial(cheops, x, y))
    return base_functions


def L_double_integral(xy_func):
    return double_integral(xy_func, -1, 0, -1, 1) + double_integral(xy_func, 0, 1, -1, 0)

def B_u_v(u_func, v_func, k):

    dudx = partial(double_derivative, u_func, "x")
    dudy = partial(double_derivative, u_func, "y")

    dvdx = partial(double_derivative, v_func, "x")
    dvdy = partial(double_derivative, v_func, "y")

    first_int = -L_double_integral(partial(lambda k, funca, funcb, x, y: k(x,y) * funca(x, y)*funcb(x, y), k, dudx, dvdx))
    second_int = -L_double_integral(partial(lambda k, funca, funcb, x, y: k(x,y) * funca(x, y)*funcb(x, y), k, dudy, dvdy))
    return first_int + second_int



def L_v(g, v):
    func = partial(lambda g, v, x, y: g(x,y)*v(x,y), g, v)
    return -(-integral(func,-1,0,1,"dx")  - integral(func,0,1,1,"dy") + integral(func, -1, 1, -1, "dx") + integral(func,-1,1,-1,"dy"))

def g_condition(x,y):
    return (x**2)**(1.0/3)

def k_condition(x,y):
    if y >= 0.5:
        return 2
    else:
        return 1

def result_func_template(base_func, x_res, x, y):
    result = 0
    for i in range(len(base_func)):
        result += base_func[i](x,y)*x_res[i]
    return result


# plotting
def plot(func):
    n = 250
    value = [[0] * n for i in range(n)]
    step = 2.0 / n
    for i in range(n):
        for j in range(n):
            value[i][j] = func(-1 + i*step, -1 + j*step)

    value_max = np.max(value)
    value_min = 0
    print( "z max")
    print( value_max)
    
    Y, X = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, value, cmap='hot', vmin=value_min, vmax=value_max)
    ax.set_title('Temperature')
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()



# xy_vertices = [(-1,1), (0,1), (-1,0), (0,0), (1,0), (-1,-1), (0,-1), (1,-1)]
xy_vertices = [(-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

base_functions = base_func_generator(xy_vertices)

print(B_u_v(base_functions[0], base_functions[1], k_condition))


row = []
col = []
for i in base_functions:
    for j in base_functions:
        row.append(B_u_v(j,i,k_condition))
    col.append(row)
    row = []

A = np.array(col)

row = []
for i in base_functions:
     row.append(L_v(g_condition, i))

B = np.array(row)

x_result = np.linalg.solve(A, B)

print(x_result)

result_func = partial(result_func_template, base_functions, x_result)

plot(result_func)