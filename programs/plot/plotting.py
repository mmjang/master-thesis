from matplotlib.font_manager import fontManager
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.utilities import round_value

np.seterr(divide='raise')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STSong']
rcParams['font.size'] = 10.5

#论文版面宽度
max_width = 5.87
normal_width = 0.8 * max_width
height = normal_width * 0.618 #golden ratio

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step(x):
    return x > 0

def relu(x):
    l = len(x)
    result = [0] * l
    for i in range(l):
        if(x[i] < 0):
            result[i] = 0
        else:
            result[i] = x[i]
    return result

#sigmoid
plt.figure(figsize = (normal_width / 2, height))
x = np.linspace(-10, 10, 1000)
plt.plot(x, sigmoid(x), 'k')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
#plt.title('Sigmoid 函数', fontsize=10.5)
plt.tight_layout()
plt.savefig('pic/sigmoid.pdf')


plt.figure(figsize = (normal_width / 2, height))
x = np.linspace(-10, 10, 1000)
plt.plot(x, step(x), 'k')
plt.xlabel('x')
plt.ylabel('step(x)')
#plt.title('阶跃函数', fontsize=10.5)
plt.tight_layout()
plt.savefig('pic/step.pdf')


plt.figure(figsize = (normal_width, height))
x = np.linspace(-10, 10, 1000)
plt.plot(x, relu(x), 'k')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
#plt.title('阶跃函数', fontsize=10.5)
plt.tight_layout()
plt.savefig('pic/relu.pdf')

########################################################################################
#fractal soil
surfacedims = (400,400)
fractalsurface = np.zeros(surfacedims, dtype=complextype)
weighting = (1, 1)
v1 = np.array([weighting[0] * (surfacedims[0]) / 2, weighting[1] * (surfacedims[1]) / 2])
R = np.random.RandomState()
A = R.randn(surfacedims[0], surfacedims[1])

sub_figure_width = (max_width * 0.95) / 2
plt.figure(figsize=(sub_figure_width, sub_figure_width))
plt.imshow(A, aspect='auto')
plt.xlabel('x')
plt.ylabel('y')
#plt.imshow(fractalsurface, aspect='auto', cmap='Gray')
plt.tight_layout()
plt.savefig('pic/rough_surface_random.pdf')

A = fftpack.fftn(A)
A = fftpack.fftshift(A)

plt.figure(figsize=(sub_figure_width, sub_figure_width))
plt.imshow(np.real(A), aspect='auto')
plt.xlabel('x')
plt.ylabel('y')
#plt.imshow(fractalsurface, aspect='auto', cmap='Gray')
plt.tight_layout()
plt.savefig('pic/rough_surface_fft.pdf')

dimension = 1.5
b = -(2 * dimension - 7) / 2
##generate fractal 
for i in range(surfacedims[0]):
    for j in range(surfacedims[1]):
        v2x = weighting[0] * i
        v2y = weighting[1] * j
        rr = ((v2x - v1[0])**2 + (v2y - v1[1])**2)**(1/2)
        if rr == 0:
            rr = 0.9
        fractalsurface[i, j] = A[i, j] * 1 / (rr**b)

plt.figure(figsize=(sub_figure_width, sub_figure_width))
plt.imshow(np.real(fractalsurface), aspect='auto')
plt.xlabel('x')
plt.ylabel('y')
#plt.imshow(fractalsurface, aspect='auto', cmap='Gray')
plt.tight_layout()
plt.savefig('pic/rough_surface_fft_fractal.pdf')
##restore
fractalsurface = fftpack.ifftshift(fractalsurface)
fractalsurface = np.real(fftpack.ifftn(fractalsurface))

plt.figure(figsize=(sub_figure_width, sub_figure_width))
plt.imshow(np.real(fractalsurface), aspect='auto')
plt.xlabel('x')
plt.ylabel('y')
#plt.imshow(fractalsurface, aspect='auto', cmap='Gray')
plt.tight_layout()
plt.savefig('pic/rough_surface_fractal.pdf')

hf = plt.figure(figsize=(normal_width, normal_width))
ha = hf.add_subplot(111, projection='3d')
ha.set_aspect('equal')
x = np.linspace(0, 1, surfacedims[0])
y = np.linspace(0, 1, surfacedims[1])
X, Y = np.meshgrid(x, y)
ha.plot_surface(X, Y, fractalsurface, cmap='coolwarm')
ha.set_xlabel('x')
ha.set_ylabel('y')
ha.set_zlabel('z')
ha.set_zlim(-0.1,0.1)
#plt.imshow(fractalsurface, aspect='auto', cmap='Gray')
plt.tight_layout()
plt.savefig('pic/rough_surface_fractal_3d.pdf')