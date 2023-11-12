import matplotlib.pyplot as plt

n_samp = [20000, 30000, 40000, 50000, 60000, 70000, 80000]
bias =  [
    -0.0022592811149505301,
     -0.0023127186185630672,
     -0.00098130403344397904,
     -0.001454317177810617,
     -0.0010522298880303108,
     8.4577284732018088e-05,
     0.00033535031692677464
     ]


bias_err =[
    0.00057247193989360334,
    0.00044126677499776946,
    0.00036742249907671957,
    0.00036499756855203731,
    0.0002761538598826286,
    0.00040555355282887015,
    0.00057526254197165603
    ]


fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(n_samp,bias,fmt='bo',yerr=bias_err,ms=10)
ax.set_xlim(0,85000)
ax.set_ylim(-0.005,0.005)
ax.set_xlabel('Number of Templates Sampled / Galaxy',size=22)
ax.set_ylabel('Multiplicative bias',size=22)
ax.axhspan(-0.001, 0.001, facecolor='grey', alpha=0.5)
ax.tick_params(axis='y', labelsize=16, width=2, length=8)
ax.tick_params(axis='x', labelsize=16, width=2, length=8)
fig.tight_layout()
fig.show()
fig.savefig('bias_samp_6d.png')
