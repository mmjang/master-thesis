from matplotlib.font_manager import fontManager
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

np.seterr(divide='raise')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['STSong']
rcParams['font.size'] = 10.5

#论文版面宽度
max_width = 5.87
normal_width = 0.8 * max_width
height = normal_width * 0.8 #golden ratio

real_s11_txt = 'programs/antenna/vivaldi_real_s11.txt'
gprmx_s11_txt = 'programs/antenna/gprmax_vivaldi_s11.txt'

mat_real = np.loadtxt(real_s11_txt)
mat_gprmax = np.loadtxt(gprmx_s11_txt)

plt.figure(figsize = (normal_width, height))
plt.plot(mat_real[:,0], mat_real[:,1],'k',label='实测')
plt.plot(mat_gprmax[:,0], mat_gprmax[:,1],'r',label='仿真')
plt.xlim([0.2, 2])
plt.ylim([-35, 10])
plt.xlabel('频率(GHz)')
plt.ylabel('S11(db)')
plt.legend()
plt.tight_layout()
plt.savefig('pic/vivaldi_s11_compare.pdf')