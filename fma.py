import numpy as np
import pylab as plt
from scipy.interpolate import griddata
from PyNAFF import naff
from resonance_lines import resonance_lines

x=np.load('x.npy')
y=np.load('y.npy')

Qx1, Qx = [], []
Qy1, Qy = [], []

for i in range(len(x)):
    try:
        Qx1.append(naff(x[i]-np.mean(x[i]), turns=600)[0][1])
    except:
        Qx1.append(np.nan)
    try:
        Qy1.append(naff(y[i]-np.mean(y[i]), turns=600)[0][1])
    except:
        Qy1.append(np.nan)
    try:
        Qx.append(naff(x[i]-np.mean(x[i]), skipTurns=599, turns=600)[0][1])
    except:
        Qx.append(np.nan)
    try:
        Qy.append(naff(y[i]-np.mean(y[i]), skipTurns=599, turns=600)[0][1])
    except:
        Qy.append(np.nan)
Qx1=4.0+np.array(Qx1)
Qx=4.0+np.array(Qx)
Qy1=4.0+np.array(Qy1)
Qy=4.0+np.array(Qy)
d = np.log(np.sqrt( (Qx-Qx1)**2 + (Qy-Qy1)**2 ))

tune_diagram=resonance_lines([3.9,4.3],[3.9,4.45], [1,2,3,4], 16)
fig=tune_diagram.plot_resonance()
plt.title('Tune Diagram', fontsize='20')
plt.scatter(Qx, Qy,4, d, 'o',lw = 0.1,zorder=10, cmap=plt.cm.jet)
plt.plot([4.2],[4.4],'ko',zorder=1e5)
plt.xlabel('$\mathrm{Q_x}$', fontsize='20')
plt.ylabel('$\mathrm{Q_y}$', fontsize='20')
plt.tick_params(axis='both', labelsize='18')
plt.clim(-20.5,-4.5)
cbar=plt.colorbar()
cbar.set_label('d',fontsize='18')
cbar.ax.tick_params(labelsize='18')
plt.tight_layout()
plt.savefig('test_fma.png')

fig2=plt.figure()
XX,YY = np.meshgrid(np.unique(x[:,0]), np.unique(y[:,0]))
Z = griddata((x[:,0],y[:,0]), d, (XX,YY), method='linear')
Zm = np.ma.masked_invalid(Z)
fig2.suptitle('Initial Distribution', fontsize='20')
plt.pcolormesh(XX,YY,Zm,cmap=plt.cm.jet)
# plt.scatter(x[:,0],y[:,0],4, d, 'o',lw = 0.1,zorder=10, cmap=plt.cm.jet)
plt.tick_params(axis='both', labelsize='18')
plt.xlabel('x [m]', fontsize='20')
plt.ylabel('y [m]', fontsize='20')
plt.clim(-20.5,-4.5)
cbar=plt.colorbar()
cbar.set_label('d',fontsize='18')
cbar.ax.tick_params(labelsize='18')
fig2.savefig('test_initial_distribution.png')


plt.show()