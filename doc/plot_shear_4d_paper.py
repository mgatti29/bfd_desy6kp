import matplotlib.pyplot as plt

shear = [0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]
bias = [
    -0.00055856129068769522,
     0.00062765112643370846,
     0.0019027421101451905,
     0.0038133277961087397,
     0.0085926694649290526,
     0.011519574125702647,
     0.018306306167773095
     ]

bias_err = [
    0.00028530692162290569,
    0.00019031579374002711,
    0.00014286987248944945,
    0.00011449762181175962,
    8.209707351330632e-05,
    7.2001402262181265e-05,
    5.7936629346782901e-05
    ]


fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(shear,bias,fmt='bo',yerr=bias_err,ms=10)
ax.set_xlabel('Input shear',size=22)
ax.set_ylabel('Multiplicative bias',size=22)
ax.set_xlim(0,0.11)
ax.axhspan(-0.001, 0.001, facecolor='grey', alpha=0.5)
ax.tick_params(axis='y', labelsize=16, width=2, length=8)
ax.tick_params(axis='x', labelsize=16, width=2, length=8)
fig.tight_layout()

import numpy as np
# include point at 0,0
shear2=[0]
bias2=[0]
shear2.extend(shear)
bias2.extend(bias)

# solve for quadratic term with constant offset
# store shear^2
shear2=[a*a for a in shear2]

A = np.vstack([shear2,np.ones(len(shear2))]).T
g = np.linalg.lstsq(A, bias2)[0]

# calculate model for a finer grid of points
xp = np.linspace(0, 0.11, 100)
x2 = [b*b for b in xp]
A2 = np.vstack([x2,np.ones(len(x2))]).T
bias_model = np.inner(g,A2)

ax.plot(xp,bias_model,'--',linewidth=2)
fig.show()
fig.savefig('bias_shear_4d.png')
