from matplotlib.font_manager import fontManager
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

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