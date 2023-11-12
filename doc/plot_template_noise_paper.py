import matplotlib.pyplot as plt

noise_frac = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
bias = [
    -0.00065566893197328141,
     0.00066439581044245438,
     0.00030235342684085953,
     0.0020464092605660539,
     0.0026559451341534247,
     0.0027880442192691414,
     0.0035964489389569976
     ]
bias_err = [
    0.00053374417823283187,
    0.00052615513773902553,
    0.00053745783898997892,
    0.00059548332582844707,
    0.00057886592317291552,
    0.00060056969472773445,
    0.00057853623828690252
    ]

fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(noise_frac,bias,fmt='bo',yerr=bias_err,ms=10)
ax.set_xlim(-0.0,0.4)
ax.set_ylim(-0.005,0.005)
ax.set_xlabel('Fractional variance in templates relative to targets',size=22)
ax.set_ylabel('Multiplicative bias',size=22)
ax.axhspan(-0.001, 0.001, facecolor='grey', alpha=0.5)
ax.tick_params(axis='y', labelsize=16, width=2, length=8)
ax.tick_params(axis='x', labelsize=16, width=2, length=8)
fig.tight_layout()
fig.show()
fig.savefig('bias_template_noise.png')
