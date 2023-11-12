import matplotlib.pyplot as plt

n_temp = [10000, 20000, 30000, 40000]
bias = [
    -0.0017387288228087883,
     -0.0014126924956234591,
     -0.00036497908886181513,
     8.4577284732018088e-05
     ]

bias_err = [
    0.00057295136891391042,
    0.00040537060262106644,
    0.00040550068045596175,
    0.00040555355282887015
    ]


fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(n_temp,bias,fmt='bo',yerr=bias_err,ms=10)
ax.set_xlim(0000,45000)
ax.set_ylim(-0.005,0.005)
ax.set_xlabel('Number of Templates',size=22)
ax.set_ylabel('Multiplicative bias',size=22)
ax.axhspan(-0.001, 0.001, facecolor='grey', alpha=0.5)
ax.tick_params(axis='y', labelsize=16, width=2, length=8)
ax.tick_params(axis='x', labelsize=16, width=2, length=8)
fig.tight_layout()
fig.show()
fig.savefig('bias_nTemp_6d.png')
