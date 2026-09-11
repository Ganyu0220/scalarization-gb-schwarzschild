import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Figure 1: lambda^2_crit vs l
# ============================================================
l_vals = np.array([0, 1, 2, 3, 4])
lam2_crit = np.array([
    3.9560787828676416,
    9.521377472089515,
    17.963655850564667,
    62.7072279284218,
    193.66707432115965,
])
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(l_vals, lam2_crit, 'o-', color='navy', markersize=8,
         linewidth=1.5, label=r'$\lambda^2_{\rm crit}(\ell)$')
ax1.axhline(10.0 / 3.0, color='red', linestyle='--', linewidth=1.2,
            label=r'$10/3$ (Doneva-Yazadjiev bound)')
ax1.set_xlabel(r'$\ell$', fontsize=13)
ax1.set_ylabel(r'$\lambda^2_{\rm crit} / M^2$', fontsize=13)
ax1.set_title(r'Scalarization thresholds: $\lambda^2_{\rm crit}$ vs $\ell$',
              fontsize=12)
ax1.set_xticks(l_vals)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('figure1_thresholds_vs_l.png', dpi=200)
print("Saved figure1_thresholds_vs_l.png")
# ============================================================
# Figure 2: c_2 extrapolation
# ============================================================
chi2_data = np.array([0.0001, 0.0004, 0.0009, 0.0025])
c2_data = np.array([
    1.5713450415182268,
    1.5720742693613876,
    1.573293816614118,
    1.5772111113177099,
])
coeffs = np.polyfit(chi2_data, c2_data, 1)
slope = coeffs[0]
intercept = coeffs[1]
chi2_fit = np.linspace(0, 0.003, 100)
c2_fit = slope * chi2_fit + intercept
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(chi2_data, c2_data, 's', color='darkgreen', markersize=9,
         label=r'Numerical $c_2(\chi^2)$')
ax2.plot(chi2_fit, c2_fit, '-', color='red', linewidth=1.2,
         label='Linear fit: slope = ' + str(round(slope, 3)))
ax2.plot(0, intercept, '*', color='gold', markersize=18,
         markeredgecolor='black', markeredgewidth=0.8,
         label='Extrapolated c2 = ' + str(round(intercept, 3)))
ax2.set_xlabel(r'$\chi^2$', fontsize=13)
ax2.set_ylabel(r'$c_2$', fontsize=13)
ax2.set_title(r'Extrapolation of $c_2$ to $\chi^2 \to 0$',
              fontsize=12)
ax2.set_xlim(-0.0002, 0.0030)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('figure2_c2_extrapolation.png', dpi=200)
print("Saved figure2_c2_extrapolation.png")
# ============================================================
# Figure 3: lambda^2_crit(chi) curve
# ============================================================
c2_val = intercept
lam2_0 = 3.9560787828676416
chi_arr = np.linspace(0, 0.5, 100)
lam2_chi = lam2_0 * (1.0 + c2_val * chi_arr ** 2)
fig3, ax3 = plt.subplots(figsize=(6, 4.5))
ax3.plot(chi_arr, lam2_chi, '-', color='purple', linewidth=2,
         label=r'$\lambda^2_{\rm crit}(\chi)$')
ax3.axhline(10.0 / 3.0, color='red', linestyle='--', linewidth=1.0,
            label=r'$10/3$ bound')
ax3.plot([0.1], [lam2_0 * (1 + c2_val * 0.01)], 'o',
         color='blue', markersize=8)
ax3.plot([0.3], [lam2_0 * (1 + c2_val * 0.09)], 'o',
         color='blue', markersize=8)
ax3.set_xlabel(r'$\chi$', fontsize=13)
ax3.set_ylabel(r'$\lambda^2_{\rm crit}(\chi)/M^2$', fontsize=13)
ax3.set_title(r'Rotation correction to the $\ell=0$ threshold',
              fontsize=12)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig('figure3_threshold_vs_chi.png', dpi=200)
print("Saved figure3_threshold_vs_chi.png")
print()
print("=== All figures saved ===")
print("c_2 (extrapolated) = " + str(intercept))
print("slope = " + str(slope))
