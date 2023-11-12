import matplotlib.pyplot as plt

sigma_shears = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
sigma_bias = [
    -0.00097697228593589258,
     -9.5227306606653073e-05,
     0.0013435164892208404,
     0.0014530606076996327,
     0.0035310846899389357,
     0.0046515532746472382
     ]
sigma_bias_err =[
    0.00057185660173095687,
    0.00057642025445133675,
    0.00058526299234136394,
    0.00060245859194408028,
    0.00062760128219044125,
    0.00084431036609674561
    ]

const_shears = [0.013, 0.025, 0.05, 0.075, 0.1]
const_bias = [
    -0.00088050979460494588,
     -0.00020113255527528379,
     0.0017602217420382288,
     0.004342347060305729,
     0.0069443258235161306
     ]
const_bias_err = [
    0.00057102181315171535,
    0.00057591811933894186,
    0.00059306054041121747,
    0.00062234103954490717,
    0.00066459489771671435
    ]





fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(sigma_shears,sigma_bias,fmt='bo',yerr=sigma_bias_err,label='Sigma width', ms=10)
ax.errorbar(const_shears,const_bias,fmt='bo',yerr=const_bias_err,label='Constant',color='red',ms=10)
ax.set_xlim(0,0.11)
ax.set_ylim(-0.005,0.005)
ax.set_xlabel('Template Shear',size=22)
ax.set_ylabel('Multiplicative bias',size=22)
ax.axhspan(-0.001, 0.001, facecolor='grey', alpha=0.5)
ax.tick_params(axis='y', labelsize=16, width=2, length=8)
ax.tick_params(axis='x', labelsize=16, width=2, length=8)
ax.legend(loc='upper left')
fig.tight_layout()
fig.show()
fig.savefig('bias_template_shear.png')

