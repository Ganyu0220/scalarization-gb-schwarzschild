import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def f_r(r):
    return 1.0 - 2.0 / r


def Veff(r, lam2, l):
    return f_r(r) * (2.0 / r ** 3 + l * (l + 1) / r ** 2 - 12.0 * lam2 / r ** 6)


def r_of_rstar(rstar):
    if rstar > 50:
        return rstar
    r = max(2.001, 2.0 + np.exp((rstar - 2.0) / 2.0))
    for _ in range(50):
        g = r + 2.0 * np.log(r / 2.0 - 1.0) - rstar
    gp = 1.0 + 2.0 / (r - 2.0)
    r = r - g / gp
    r = max(2.001, r)
    return r


def rhs(rstar, y, lam2, l):
    uu, up = y
    r = r_of_rstar(rstar)
    v = Veff(r, lam2, l)
    return [up, v * uu]


def shoot(lam2, l):
    sol = solve_ivp(rhs, [-30.0, 30.0], [0.0, 1.0],
                    args=(lam2, l), rtol=1e-10, atol=1e-12)
    return sol.y[0][-1]


def find_crit(scan_values, l):
    values = [shoot(x, l) for x in scan_values]
    for i in range(len(scan_values) - 1):
        if values[i] * values[i + 1] < 0:
            crit = brentq(lambda x: shoot(x, l),
                          scan_values[i], scan_values[i + 1], xtol=1e-10)
    return crit
    return None


# ============================================================
# l = 1
# ============================================================print("=== Scan for l=1 (range 3 to 20) ===")
lam2_scan_l1 = np.linspace(3.0, 20.0, 40)
for e in lam2_scan_l1:
    v = shoot(e, 1)
    print(str(round(e, 3)) + " " + str(v))
print()
lam2_c1 = find_crit(lam2_scan_l1, 1)
if lam2_c1 is None:
    print("l=1: no sign change found in [3, 20].")
else:
    print("l=1 critical: lambda^2 = " + str(lam2_c1))
    print(" lambda = " + str(np.sqrt(lam2_c1)))
# ============================================================
# l = 3
# ============================================================
print()
print("=== Scan for l=3 (range 18 to 100) ===")
lam2_scan_l3 = np.linspace(18.0, 100.0, 40)
for e in lam2_scan_l3:
    v = shoot(e, 3)
    print(str(round(e, 3)) + " " + str(v))
print()
lam2_c3 = find_crit(lam2_scan_l3, 3)
if lam2_c3 is None:
    print("l=3: no sign change found in [18, 100].")
else:
    print("l=3 critical: lambda^2 = " + str(lam2_c3))
    print(" lambda = " + str(np.sqrt(lam2_c3)))
# ============================================================
# l = 4
# ============================================================
print()
print("=== Scan for l=4 (range 18 to 200) ===")
lam2_scan_l4 = np.linspace(18.0, 200.0, 40)
for e in lam2_scan_l4:
    v = shoot(e, 4)
    print(str(round(e, 3)) + " " + str(v))
    print()
lam2_c4 = find_crit(lam2_scan_l4, 4)
if lam2_c4 is None:
    print("l=4: no sign change found in [18, 200].")
else:
    print("l=4 critical: lambda^2 = " + str(lam2_c4))
    print(" lambda = " + str(np.sqrt(lam2_c4)))
# ============================================================
# Summary table
# ============================================================
print()
print("=== Summary: l vs lambda^2_crit ===")
# l=0 and l=2 from previous runs (hardcoded for completeness)
results = {
    0: 3.9560787828676416,
    1: lam2_c1,
    2: 17.963655850564667,
    3: lam2_c3,
    4: lam2_c4,
}
for l_val in range(5):
    val = results[l_val]
    if val is None:
        print("l=" + str(l_val) + ": (not found)")
    else:
        print("l=" + str(l_val) + ": lambda^2_crit = " + str(val) +
              " lambda_crit = " + str(np.sqrt(val)))
print()
print("=== Done ===")
print()
print("=== Rotation correction: l=0, chi^2 shift ===")


def Veff_l0_rot(r, lam2, chi2):
    # U_0(r) = f(r) * [2/r^3 - 12 lam2 / r^6 + 100 lam2 * chi2 / r^8]
    # (M = 1)
    base = 2.0 / r ** 3 - 12.0 * lam2 / r ** 6
    rot = 100.0 * lam2 * chi2 / r ** 8
    return f_r(r) * (base + rot)


def rhs_l0_rot(rstar, y, lam2, chi2):
    uu, up = y
    r = r_of_rstar(rstar)
    v = Veff_l0_rot(r, lam2, chi2)
    return [up, v * uu]


def shoot_l0_rot(lam2, chi2):
    sol = solve_ivp(rhs_l0_rot, [-30.0, 30.0], [0.0, 1.0],
                    args=(lam2, chi2), rtol=1e-10, atol=1e-12)
    return sol.y[0][-1]


def find_crit_rot(scan_values, chi2):
    values = [shoot_l0_rot(x, chi2) for x in scan_values]
    for i in range(len(scan_values) - 1):
        if values[i] * values[i + 1] < 0:
            crit = brentq(lambda x: shoot_l0_rot(x, chi2),
                          scan_values[i], scan_values[i + 1], xtol=1e-10)
    return crit
    return None


# Baseline: chi = 0 threshold was 3.9560787828676416
# Test chi^2 = 0.01 (chi ~ 0.1), 0.04 (chi ~ 0.2), 0.09 (chi ~ 0.3)
print("chi^2 lambda^2_crit c_2 estimate")
lam2_crit_chi0 = 3.9560787828676416
for chi2_val in [0.01, 0.04, 0.09]:
    scan = np.linspace(3.5, 5.0, 40)
    crit = find_crit_rot(scan, chi2_val)
    if crit is None:
        print(str(chi2_val) + " (no sign change)")
    else:
        c2 = (crit - lam2_crit_chi0) / (lam2_crit_chi0 * chi2_val)
    print(str(chi2_val) + " " + str(crit) + " " + str(c2))
print()
print("=== Done ===")
print()
print("=== c_2 extraction with smaller chi^2 values ===")
print("chi^2 lambda^2_crit c_2 estimate")
lam2_crit_chi0 = 3.9560787828676416
for chi2_val in [0.0001, 0.0004, 0.0009, 0.0025]:
    scan = np.linspace(3.95, 4.05, 40)
    crit = find_crit_rot(scan, chi2_val)
    if crit is None:
        print(str(chi2_val) + " (no sign change)")
    else:
        c2 = (crit - lam2_crit_chi0) / (lam2_crit_chi0 * chi2_val)
    print(str(chi2_val) + " " + str(crit) + " " + str(c2))
print()
print("=== Done ===")
