#!/usr/bin/env python3
"""
replication.py  --  Reproduce all results from:

  "A Riemannian Metric for Three-Dimensional Color Discrimination
   Derived from V1 Population Fisher Information"

This script evaluates the 17-parameter biological model and four
standard color-difference formulas on four threshold datasets:
MacAdam (1942), Koenderink et al. (2026), Wright (1941), and
Huang et al. (2012).

Sections:
  1. Joint-fit STRESS and diagnostics (Table 1)
  2. Standard metric comparison (CIELAB, CIEDE2000, CIECAM02-UCS, CAM16-UCS)
  3. Huang per-centre breakdown
  4. Ablation studies (eta=0, delta_nc=0, tau->inf)

All data is hardcoded.  No network access or external files required.

Requirements:
    numpy, scipy
    colour-science  (only for CIEDE2000 / CIECAM02-UCS / CAM16-UCS)

Usage:
    python3 replication.py
"""

import sys
import numpy as np


# =====================================================================
#  Smith-Pokorny cone fundamentals
#  Transforms CIE 1931 XYZ to LMS cone excitations.
#  Source: Smith & Pokorny (1975), via Bowmaker & Dartnall (1980).
# =====================================================================

M_SP = np.array([
    [ 0.15514,  0.54312, -0.03286],
    [-0.15514,  0.45684,  0.03286],
    [ 0.0,      0.0,      0.01608],
])


# =====================================================================
#  sRGB display model  (for the Koenderink data, reported in sRGB)
#  Peak luminance Y_WHITE = 214 cd/m^2 as measured by the authors.
# =====================================================================

_M_sRGB_UNIT = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

Y_WHITE = 214.0
M_sRGB = _M_sRGB_UNIT * Y_WHITE


def srgb_gamma_decode(v):
    """sRGB EOTF: gamma-encoded [0,1] to linear [0,1]."""
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def rgb_to_xyY(rgb):
    """Gamma-encoded sRGB [0,1]^3 to CIE (x, y, Y) in cd/m^2."""
    rgb_lin = np.array([srgb_gamma_decode(c) for c in rgb])
    XYZ = M_sRGB @ rgb_lin
    s = XYZ[0] + XYZ[1] + XYZ[2]
    if s < 1e-10:
        return np.array([0.3127, 0.3290, 1e-6])
    return np.array([XYZ[0] / s, XYZ[1] / s, XYZ[1]])


# =====================================================================
#  Coordinate conversions
# =====================================================================

def xyY_to_XYZ(x, y, Y):
    if y < 1e-10:
        return np.array([0.0, Y, 0.0])
    return np.array([Y * x / y, Y, Y * (1.0 - x - y) / y])


def xyY_to_LMS(x, y, Y):
    return np.maximum(M_SP @ xyY_to_XYZ(x, y, Y), 1e-12)


# CIELAB conversions (10-deg D65 white for Huang dataset)
XN_10, YN_10, ZN_10 = 94.811, 100.0, 107.304


def lab_to_xyY(L, a, b):
    """CIELAB (L*, a*, b*) with 10-deg D65 white to CIE (x, y, Y)."""
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t):
        return t ** 3 if t > d else 3 * d * d * (t - 4.0 / 29.0)

    X = XN_10 * finv(fx)
    Y = YN_10 * finv(fy)
    Z = ZN_10 * finv(fz)
    s = X + Y + Z
    if s < 1e-10:
        return 0.3127, 0.3290, 0.01
    return X / s, Y / s, Y


def xyY_to_lab(x, y, Yv):
    """CIE (x, y, Y) to CIELAB (L*, a*, b*) with 10-deg D65 white."""
    if y < 1e-10:
        return 0.0, 0.0, 0.0
    X = Yv * x / y
    Z = Yv * (1.0 - x - y) / y
    d3 = (6.0 / 29.0) ** 3

    def fwd(t):
        return t ** (1.0 / 3.0) if t > d3 else t / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0

    L = 116.0 * fwd(Yv / YN_10) - 16.0
    a = 500.0 * (fwd(X / XN_10) - fwd(Yv / YN_10))
    b = 200.0 * (fwd(Yv / YN_10) - fwd(Z / ZN_10))
    return L, a, b


# =====================================================================
#  Adaptation whites
#  MacAdam and Wright used Illuminant C.
#  Koenderink and Huang used D65.
# =====================================================================

ILLC_xy = (0.31006, 0.31616)
D65_xy  = (0.31272, 0.32903)
Y_REF   = 48.0

LMS_ILLC = xyY_to_LMS(ILLC_xy[0], ILLC_xy[1], Y_REF)
LMS_D65  = xyY_to_LMS(D65_xy[0],  D65_xy[1],  Y_REF)


# =====================================================================
#  Dataset 1:  MacAdam (1942)
#  25 threshold ellipses, observer PGN, Y = 48 cd/m^2, Illuminant C
#  Format: (x, y, a*1e3, b*1e3, theta_degrees)
#  Source: Wyszecki & Stiles (2000), Table 1(5.4.1)
# =====================================================================

MACADAM = [
    (0.160, 0.057, 0.85, 0.35, 62.5),
    (0.187, 0.118, 2.20, 0.55, 77.0),
    (0.253, 0.125, 2.50, 0.50, 55.5),
    (0.150, 0.680, 9.60, 2.30, 105.0),
    (0.131, 0.521, 4.70, 2.00, 112.5),
    (0.212, 0.550, 5.80, 2.30, 100.0),
    (0.258, 0.450, 5.00, 2.00, 92.0),
    (0.152, 0.365, 3.80, 1.90, 110.0),
    (0.280, 0.385, 4.00, 1.50, 75.5),
    (0.380, 0.498, 4.40, 1.20, 70.0),
    (0.160, 0.200, 2.10, 0.95, 104.0),
    (0.228, 0.250, 3.10, 0.90, 72.0),
    (0.305, 0.323, 2.30, 0.90, 58.0),
    (0.385, 0.393, 3.80, 1.60, 65.5),
    (0.472, 0.399, 3.20, 1.40, 51.0),
    (0.527, 0.350, 2.60, 1.30, 20.0),
    (0.475, 0.300, 2.90, 1.10, 28.5),
    (0.510, 0.236, 2.40, 1.20, 29.5),
    (0.596, 0.283, 2.60, 1.30, 13.0),
    (0.344, 0.284, 2.30, 0.90, 60.0),
    (0.390, 0.237, 2.50, 1.00, 47.0),
    (0.441, 0.198, 2.80, 0.95, 34.5),
    (0.278, 0.223, 2.40, 0.55, 57.5),
    (0.300, 0.163, 2.90, 0.60, 54.0),
    (0.365, 0.153, 3.60, 0.95, 40.0),
]


# =====================================================================
#  Dataset 2:  Koenderink, van Doorn, Braun & Gegenfurtner (2026)
#  35 three-dimensional ellipsoids, 8 observers (group median), D65
#  Format: (R*1000, G*1000, B*1000,  Sig11, Sig12, Sig13, Sig22, Sig23, Sig33)
#     RGB values are gamma-encoded sRGB * 1000.
#     Sigma_ij values are scaled by 1e7.
#  Source: bioRxiv 2026.03.09.710376, Appendix A2
# =====================================================================

KOENDERINK = [
    (200,200,200, 5921, 1948, 928, 3886, 1021, 4749),
    (200,200,500, 9215, 1360, 4103, 11861, 1386, 15525),
    (200,200,800, 21690,-3347,10627, 48696,-3215, 36789),
    (200,500,200, 35342, 1900, 2635, 14007, 3373, 27777),
    (200,500,500, 24694, 3463, 1482, 12425, 5167, 14643),
    (200,500,800, 26625, 1302, 5918, 17913, 3172, 32615),
    (200,800,200,165975, 5955, 2116, 28389, 6863, 41825),
    (200,800,500, 73111, 9497, 5039, 28421, 9518, 23966),
    (200,800,800,101190,-2606, 6015, 16789, 6767, 20554),
    (350,350,350, 6783, 3291, 2856, 8216, 3095, 10971),
    (350,350,650, 9701, 3105, 749, 12794, 742, 22673),
    (350,650,350, 25572, 6261, 8090, 18852, 5571, 25738),
    (350,650,650, 23178, 6578, 6647, 21049, 8966, 17971),
    (500,200,200, 14834, 2552, 2079, 10992, 4923, 12283),
    (500,200,500, 15568, 3162, 4692, 20208, 3389, 20117),
    (500,200,800, 14308, 3288, 7870, 22941, -489, 20016),
    (500,500,200, 15405, 6395, 1161, 13440, 4827, 23050),
    (500,500,500, 9587, 5295, 4776, 8469, 6427, 14371),
    (500,500,800, 15252, 4851, 7727, 13422, 2622, 25493),
    (500,800,200, 37494,12352, 8210, 32347, 7589, 39717),
    (500,800,500, 19748, 5502, 2284, 22837, 8286, 19868),
    (500,800,800, 22456, 7329, 6247, 21250, 9843, 21454),
    (650,350,350, 15701, 4312, 2756, 12205, 5799, 11475),
    (650,350,650, 16335, 5107, 6724, 19765, 157, 26332),
    (650,650,350, 19619, 8583, 7085, 15874, 5627, 18256),
    (650,650,650, 17727,13937,11284, 17931,12472, 19410),
    (800,200,200, 28293, 4685, 5244, 23305, 6907, 16204),
    (800,200,500, 22889, 6518,12858, 35380, 6847, 26719),
    (800,200,800, 26613, 4662, 9539, 64125, -556, 44888),
    (800,500,200, 19053, 6333, 6568, 16388, 5754, 20538),
    (800,500,500, 22663, 9771, 8971, 17408, 9786, 20296),
    (800,500,800, 21846, 6681,11800, 20108, 6351, 31972),
    (800,800,200, 23604,13996, 500, 21787, -739, 32105),
    (800,800,500, 19239, 8751, 9896, 19582,11909, 23151),
    (800,800,800, 19382, 6859, 7526, 14190, 7048, 21049),
]


# =====================================================================
#  Dataset 3:  Wright (1941)
#  19 wavelength discrimination thresholds along the spectrum locus.
#  Single observer, Y = 48 cd/m^2, Illuminant C.
#  Format: (lambda_nm, delta_lambda_nm,
#           x, y, dx/dlambda, dy/dlambda)
#  Chromaticity gradients computed from CIE 1931 2-deg CMFs at 0.5 nm
#  spacing using central differences.
# =====================================================================

WRIGHT = [
    (440, 1.2, 0.164412, 0.010858, -0.00056903, 0.00051424),
    (450, 1.2, 0.156641, 0.017705, -0.00100613, 0.00088949),
    (460, 1.4, 0.143960, 0.029703, -0.00151315, 0.00166563),
    (470, 2.0, 0.124118, 0.057803, -0.00253487, 0.00450856),
    (480, 2.8, 0.091294, 0.132702, -0.00406274, 0.01112883),
    (490, 3.5, 0.045391, 0.294976, -0.00471215, 0.02163648),
    (500, 4.0, 0.008168, 0.538423, -0.00203536, 0.02509784),
    (510, 3.2, 0.013870, 0.750186,  0.00369386, 0.01573790),
    (520, 2.0, 0.074302, 0.833803,  0.00759354, 0.00109410),
    (530, 1.5, 0.154722, 0.805864,  0.00781411,-0.00447180),
    (540, 1.2, 0.229620, 0.754329,  0.00726820,-0.00573410),
    (550, 1.1, 0.301604, 0.692308,  0.00714941,-0.00654671),
    (560, 1.0, 0.373102, 0.624451,  0.00711696,-0.00690661),
    (570, 1.1, 0.444062, 0.554714,  0.00697792,-0.00691535),
    (580, 1.5, 0.512486, 0.486591,  0.00656673,-0.00654246),
    (590, 2.5, 0.575151, 0.424232,  0.00578820,-0.00576730),
    (600, 4.0, 0.627037, 0.372491,  0.00451316,-0.00449343),
    (610, 4.5, 0.665764, 0.334011,  0.00318715,-0.00316681),
    (620, 3.5, 0.691504, 0.308342,  0.00206173,-0.00205418),
]


# =====================================================================
#  Dataset 4:  Huang, Cui, Melgosa, Sanchez-Maranon, Li, Luo, Liu (2012)
#  17 colour centres with threshold ellipses in CIELAB a*b* plane.
#  10-deg observer, D65, varied luminance and chromaticity.
#  Format: (L*, a*, b*, semi-major axis A, axis ratio A/B, theta_deg)
#  Source: Color Res. Appl. 37(6), Table 2.
# =====================================================================

HUANG = [
    (63.2,  -0.3,    0.3, 1.25, 1.74,  64.53),
    (46.6,  37.8,   23.0, 1.92, 1.98,  60.89),
    (44.8,  58.2,   36.9, 2.27, 2.05,  46.18),
    (64.4,  13.5,   22.4, 2.13, 3.01,  82.29),
    (61.6,  34.2,   63.5, 3.71, 4.49,  86.98),
    (86.5,  -6.3,   48.0, 1.91, 1.94,  92.42),
    (86.2,  -9.5,   72.7, 2.79, 4.16,  85.90),
    (66.5, -10.6,   14.0, 1.25, 1.63,  99.53),
    (65.9, -30.4,   41.4, 3.02, 4.19, 111.58),
    (57.9, -32.5,    1.0, 1.23, 1.30, 142.72),
    (58.7, -37.8,    0.6, 1.78, 2.15, 161.37),
    (51.4, -16.7,  -10.3, 1.10, 1.54,   3.57),
    (52.6, -27.1,  -17.8, 1.31, 1.61,  61.14),
    (37.9,   5.5,  -31.8, 1.05, 1.49, 122.00),
    (36.7,   6.2,  -44.4, 1.22, 1.54, 131.51),
    (46.9,  11.5,  -13.3, 2.15, 3.40, 105.21),
    (46.7,  26.3,  -26.3, 1.96, 3.05, 128.78),
]

HUANG_NAMES = [
    "Grey", "Red", "Red HC", "Orange", "Orange HC",
    "Yellow", "Yellow HC", "Yellow-green", "YG HC",
    "Green", "Green HC", "Blue-green", "BG HC",
    "Blue", "Blue HC", "Purple", "Purple HC",
]


# =====================================================================
#  Model parameters (17p, jointly fitted)
#
#  The model was optimized with differential evolution (scipy, pop=30,
#  300 generations) using a weighted objective (1.5*MacAdam + 1.0*Koen
#  + 1.0*Wright + 0.75*Huang) followed by Nelder-Mead polishing with
#  a penalty term to enforce biological bounds.
# =====================================================================

PARAMS = np.array([
    0.596802,    #  0  p_s         S-cone transduction exponent
    8.616093,    #  1  kappa_s     S-cone Naka-Rushton half-saturation
    12.854079,   #  2  gamma       konio/parvo sensitivity ratio
    2.354558,    #  3  eta         non-cardinal V1 population weight
    -2.243597,   #  4  theta_0     non-cardinal mean direction (rad)
    0.993793,    #  5  theta_1     direction gradient along z
    1.508492,    #  6  theta_2     direction gradient along w
    1.993487,    #  7  z_max       L-M Naka-Rushton saturation ceiling
    1.021526,    #  8  p_z         L-M Naka-Rushton exponent
    1.392899,    #  9  kappa_z     L-M half-saturation
    0.716583,    # 10  a_P         parvo luminance exponent
    0.567920,    # 11  a_K         konio luminance exponent
    0.737806,    # 12  psi_0       magnocellular Weber fraction
    0.840674,    # 13  beta_Y      magnocellular luminance exponent
    0.796076,    # 14  tau         konio cross-normalization
    10.000691,   # 15  delta_nc    S-cone disinhibition strength
    0.011876,    # 16  sigma_w     disinhibition half-saturation
])

PARAM_NAMES = [
    "p_s", "kappa_s", "gamma", "eta", "theta_0", "theta_1", "theta_2",
    "z_max", "p_z", "kappa_z", "a_P", "a_K", "psi_0", "beta_Y", "tau",
    "delta_nc", "sigma_w",
]

# Numerical differentiation step size
_H = 1e-7


# =====================================================================
#  Opponent coordinates
#
#  z: parvocellular (L-M), Naka-Rushton on log(L_a / M_a)
#  w: koniocellular (S), Naka-Rushton on S_a / (L_a + M_a)
#
#  Both are luminance-invariant by construction.
# =====================================================================

def opponent_coords(x, y, Y, params, adapt_lms):
    """Compute opponent coordinates (z, w) from CIE (x, y, Y)."""
    p_s     = params[0]
    kappa_s = params[1]
    z_max   = params[7]
    p_z     = params[8]
    kappa_z = params[9]

    lms = xyY_to_LMS(x, y, Y)
    la = lms / adapt_lms
    if np.any(la < 1e-15):
        return None

    # Parvocellular: saturating compression of the log L/M ratio
    u = np.log(la[0] / la[1])
    au = abs(u)
    if au < 1e-15:
        z = 0.0
    else:
        z = z_max * np.sign(u) * au ** p_z / (au ** p_z + kappa_z)

    # Koniocellular: saturating compression of S / (L + M)
    r = la[2] / (la[0] + la[1])
    if r < 1e-15:
        return None
    rp = r ** p_s
    w = rp / (rp + kappa_s)

    return np.array([z, w])


# =====================================================================
#  Biological metric in opponent space
#
#  G = G_P + G_K + G_N
#
#  G_P = Y^{a_P} * e_z e_z^T                         (parvocellular)
#  G_K = N(z) * gamma^2 * Y^{a_K} * e_w e_w^T        (koniocellular)
#  G_N = eta^2 * S(w) * Y^{(a_P+a_K)/2} * n_hat n_hat^T  (non-cardinal)
#
#  where:
#    N(z) = tau / (tau + z^2)            (konio cross-normalization)
#    S(w) = 1 + delta * sigma^2 / (w^2 + sigma^2)   (S-cone disinhibition)
#    n_hat = (cos(theta), sin(theta))
#    theta = theta_0 + theta_1 * z + theta_2 * (w - w_0)
# =====================================================================

def bio_metric_2d(x, y, Y, params, adapt_lms):
    """
    Compute the 2x2 chromaticity metric in CIE (x, y) coordinates.

    Returns a 2x2 positive definite matrix, or None if the point
    falls outside the computable domain.
    """
    p_s      = params[0]
    kappa_s  = params[1]
    gamma    = params[2]
    eta      = params[3]
    theta_0  = params[4]
    theta_1  = params[5]
    theta_2  = params[6]
    a_P      = params[10]
    a_K      = params[11]
    tau      = params[14]
    delta_nc = params[15]
    sigma_w  = params[16]

    # Adaptation white's w coordinate
    w0 = 0.5 ** p_s / (0.5 ** p_s + kappa_s)

    def oc(cx, cy):
        return opponent_coords(cx, cy, Y, params, adapt_lms)

    zw = oc(x, y)
    if zw is None:
        return None

    # Jacobian d(z, w) / d(x, y) by central differences
    a = oc(x + _H, y)
    b = oc(x - _H, y)
    c = oc(x, y + _H)
    d = oc(x, y - _H)
    if any(v is None for v in [a, b, c, d]):
        return None
    J = np.column_stack([(a - b) / (2 * _H), (c - d) / (2 * _H)])

    z, w = zw[0], zw[1]

    # Non-cardinal preferred direction
    theta = theta_0 + theta_1 * z + theta_2 * (w - w0)
    ndir = np.array([np.cos(theta), np.sin(theta)])

    # Konio cross-normalization: suppresses S-cone pathway when L-M is active
    N = tau / (tau + z * z)

    # S-cone disinhibition: boosts non-cardinal when S-cone input is absent
    S = 1.0 + delta_nc * sigma_w ** 2 / (w ** 2 + sigma_w ** 2)

    # Assemble the 2x2 Fisher information in opponent coordinates
    Ys = max(Y, 0.01)
    G = np.diag([Ys ** a_P, N * gamma ** 2 * Ys ** a_K])
    G += eta ** 2 * S * Ys ** ((a_P + a_K) / 2.0) * np.outer(ndir, ndir)

    # Pull back to CIE (x, y) coordinates
    g = J.T @ G @ J

    if np.linalg.eigvalsh(g)[0] <= 0:
        return None
    return g


def bio_sigma_3d(entry, params):
    """
    Compute the 3x3 predicted covariance matrix in sRGB coordinates
    for a Koenderink data point.

    The block-diagonal metric in (x, y, Y) is transformed to sRGB
    via the Jacobian of the display model.
    """
    psi_0  = params[12]
    beta_Y = params[13]
    rgb = entry["rgb"]
    xyY = rgb_to_xyY(rgb)
    x0, y0, Y0 = xyY

    if Y0 < 1e-6:
        return None

    gc = bio_metric_2d(x0, y0, Y0, params, LMS_D65)
    if gc is None:
        return None

    Ys = max(Y0, 0.01)

    # Block-diagonal 3x3 metric in (x, y, Y)
    g3 = np.zeros((3, 3))
    g3[:2, :2] = gc
    g3[2, 2] = psi_0 ** 2 / Ys ** (2 * beta_Y)

    # Jacobian d(x, y, Y) / d(R_gamma, G_gamma, B_gamma) by central differences
    Jf = np.zeros((3, 3))
    for j in range(3):
        rp = rgb.copy()
        rm = rgb.copy()
        rp[j] = min(rp[j] + _H, 1.0)
        rm[j] = max(rm[j] - _H, 0.0)
        delta = rp[j] - rm[j]
        if delta < 1e-12:
            return None
        Jf[:, j] = (rgb_to_xyY(rp) - rgb_to_xyY(rm)) / delta

    if abs(np.linalg.det(Jf)) < 1e-20:
        return None

    # Transform metric to sRGB, then invert to get covariance
    gr = Jf.T @ g3 @ Jf
    if np.linalg.eigvalsh(gr)[0] <= 0:
        return None
    return np.linalg.inv(gr)


# =====================================================================
#  CIELAB metric (for comparison)
# =====================================================================

def cielab_metric_2d(x, y, Y):
    """Numerically computed 2x2 CIELAB metric at (x, y, Y)."""
    h = 1e-6
    L0, a0, b0 = xyY_to_lab(x, y, Y)

    Lxp, axp, bxp = xyY_to_lab(x + h, y, Y)
    Lxm, axm, bxm = xyY_to_lab(x - h, y, Y)
    Lyp, ayp, byp = xyY_to_lab(x, y + h, Y)
    Lym, aym, bym = xyY_to_lab(x, y - h, Y)

    J = np.array([
        [(axp - axm) / (2 * h), (ayp - aym) / (2 * h)],
        [(bxp - bxm) / (2 * h), (byp - bym) / (2 * h)],
    ])
    g = J.T @ J
    if np.linalg.eigvalsh(g)[0] <= 0:
        return None
    return g


# =====================================================================
#  Standard metrics via colour-science library (optional)
# =====================================================================

def _try_import_colour():
    try:
        import colour
        return colour
    except ImportError:
        return None


def make_cam_metrics(colour):
    """Build 2x2 metric functions for CIEDE2000, CIECAM02-UCS, CAM16-UCS."""
    from colour.difference import delta_E_CIE2000

    h = 1e-6
    wp = np.array([XN_10, YN_10, ZN_10]) / 100.0

    def _make_jab_metric(to_jab):
        def metric(x, y, Y):
            XYZ = xyY_to_XYZ(x, y, Y) / 100.0
            J0 = to_jab(XYZ)
            Jp = to_jab(xyY_to_XYZ(x + h, y, Y) / 100.0)
            Jm = to_jab(xyY_to_XYZ(x - h, y, Y) / 100.0)
            Kp = to_jab(xyY_to_XYZ(x, y + h, Y) / 100.0)
            Km = to_jab(xyY_to_XYZ(x, y - h, Y) / 100.0)
            Jac = np.column_stack([
                (Jp[1:] - Jm[1:]) / (2 * h),
                (Kp[1:] - Km[1:]) / (2 * h),
            ])
            g = Jac.T @ Jac
            if np.linalg.eigvalsh(g)[0] <= 0:
                return None
            return g
        return metric

    def ciede2000_metric(x, y, Y):
        Lab0 = np.array(xyY_to_lab(x, y, Y))
        Labxp = np.array(xyY_to_lab(x + h, y, Y))
        Labxm = np.array(xyY_to_lab(x - h, y, Y))
        Labyp = np.array(xyY_to_lab(x, y + h, Y))
        Labym = np.array(xyY_to_lab(x, y - h, Y))
        dx = delta_E_CIE2000(Lab0, Labxp) / h
        dmx = delta_E_CIE2000(Lab0, Labxm) / h
        dy = delta_E_CIE2000(Lab0, Labyp) / h
        dmy = delta_E_CIE2000(Lab0, Labym) / h
        gxx = dx * dmx
        gyy = dy * dmy
        gxy = 0.25 * ((delta_E_CIE2000(Labxp, Labyp) ** 2
                        - delta_E_CIE2000(Labxp, Labym) ** 2
                        - delta_E_CIE2000(Labxm, Labyp) ** 2
                        + delta_E_CIE2000(Labxm, Labym) ** 2)
                       / (4 * h * h))
        g = np.array([[gxx, gxy], [gxy, gyy]])
        if np.linalg.eigvalsh(g)[0] <= 0:
            return None
        return g

    specs02 = colour.CAM_Specification_CIECAM02
    specs16 = colour.CAM_Specification_CAM16

    def ciecam02_jab(XYZ):
        spec = colour.appearance.ciecam02.CIECAM02_to_XYZ(
            XYZ_w=wp * 100, L_A=64, Y_b=20, surround=colour.VIEWING_CONDITIONS_CIECAM02['Average'],
            XYZ=XYZ * 100)
        # Actually need forward
        spec = colour.XYZ_to_CIECAM02(XYZ * 100, wp * 100, 64, 20)
        Jab = colour.JMh_CIECAM02_to_UCS_Luo2006(np.array([spec.J, spec.M, spec.h]))
        return Jab

    def cam16_jab(XYZ):
        spec = colour.XYZ_to_CAM16(XYZ * 100, wp * 100, 64, 20)
        Jab = colour.JMh_CAM16_to_UCS_Li2017(np.array([spec.J, spec.M, spec.h]))
        return Jab

    return {
        "CIEDE2000": ciede2000_metric,
        "CIECAM02-UCS": _make_jab_metric(ciecam02_jab),
        "CAM16-UCS": _make_jab_metric(cam16_jab),
    }


# =====================================================================
#  STRESS computation
# =====================================================================

def stress(obs, pred):
    """Standardized Residual Sum of Squares (STRESS) metric."""
    o = np.asarray(obs)
    p = np.asarray(pred)
    F = np.dot(o, p) / np.dot(p, p)
    return 100.0 * np.sqrt(np.sum((o - F * p) ** 2) / np.sum(o ** 2))


def ellipse_to_metric(a, b, theta_deg):
    """Convert ellipse (semi-major a, semi-minor b, angle) to 2x2 metric."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([1.0 / a ** 2, 1.0 / b ** 2]) @ R.T


def load_macadam():
    """Parse MacAdam data into dictionaries with observed metric tensors."""
    data = []
    for x, y, a3, b3, th in MACADAM:
        g_obs = ellipse_to_metric(a3 * 1e-3, b3 * 1e-3, th)
        data.append({"x": x, "y": y, "Y": Y_REF, "g_obs": g_obs})
    return data


def load_koenderink():
    """Parse Koenderink data into dictionaries with observed covariances."""
    data = []
    for row in KOENDERINK:
        rgb = np.array(row[:3], dtype=float) / 1000.0
        s11, s12, s13, s22, s23, s33 = [v / 1e7 for v in row[3:]]
        Sigma = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
        xyY = rgb_to_xyY(rgb)
        data.append({
            "rgb": rgb.copy(), "Sigma": Sigma,
            "x": xyY[0], "y": xyY[1], "Y": xyY[2],
        })
    return data


def stress_2d(data, metric_fn, n_angles=12):
    """STRESS for 2D chromaticity ellipses."""
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    obs, pred = [], []
    for ell in data:
        g = metric_fn(ell["x"], ell["y"], ell["Y"])
        if g is None:
            continue
        for a in angles:
            v = np.array([np.cos(a), np.sin(a)])
            ro = v @ ell["g_obs"] @ v
            rp = v @ g @ v
            if ro > 0 and rp > 0:
                obs.append(1.0 / np.sqrt(ro))
                pred.append(1.0 / np.sqrt(rp))
    if len(obs) < 10:
        return 999.0
    return stress(obs, pred)


def _fibonacci_directions(n=26):
    """Generate n approximately uniform directions on the unit sphere."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    dirs = []
    for i in range(n):
        theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
        phi = 2.0 * np.pi * i / golden
        dirs.append(np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]))
    return dirs

_FIBO_DIRS = _fibonacci_directions()


def stress_3d(data, sigma_fn):
    """STRESS for 3D ellipsoids, probed along Fibonacci directions."""
    obs, pred = [], []
    for ell in data:
        Sp = sigma_fn(ell)
        if Sp is None:
            continue
        for d in _FIBO_DIRS:
            ro = np.sqrt(d @ ell["Sigma"] @ d)
            rp = np.sqrt(d @ Sp @ d)
            if ro > 0 and rp > 0:
                obs.append(ro)
                pred.append(rp)
    if len(obs) < 10:
        return 999.0
    return stress(obs, pred)


def stress_wright(wright_data, metric_fn, Y=48.0):
    """STRESS for Wright wavelength discrimination thresholds."""
    obs, pred = [], []
    for lam, dlam, x, y, dxdl, dydl in wright_data:
        g = metric_fn(x, y, Y)
        if g is None:
            continue
        v = np.array([dxdl, dydl])
        s = v @ g @ v
        if s > 0:
            obs.append(dlam)
            pred.append(1.0 / np.sqrt(s))
    if len(obs) < 5:
        return 999.0
    return stress(obs, pred)


def stress_huang(huang_data, metric_fn_adapt, adapt_lms, n_angles=12):
    """STRESS for Huang CIELAB ellipses, transformed to CIE (x, y)."""
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    h = 1e-5
    obs, pred = [], []

    for L, al, bl, A, AB_ratio, theta_deg in huang_data:
        x0, y0, Y0 = lab_to_xyY(L, al, bl)
        if y0 < 1e-6 or Y0 < 0.1:
            continue

        g_pred = metric_fn_adapt(x0, y0, Y0, adapt_lms)
        if g_pred is None:
            continue

        # Observed metric in (a*, b*)
        B = A / AB_ratio
        theta = theta_deg * np.pi / 180.0
        ct, st = np.cos(theta), np.sin(theta)
        iA = 1.0 / (A * A)
        iB = 1.0 / (B * B)
        G_ab = np.array([
            [ct * ct * iA + st * st * iB, ct * st * (iA - iB)],
            [ct * st * (iA - iB),          st * st * iA + ct * ct * iB],
        ])

        # Jacobian d(a*, b*) / d(x, y) by central differences
        _, axp, bxp = xyY_to_lab(x0 + h, y0, Y0)
        _, axm, bxm = xyY_to_lab(x0 - h, y0, Y0)
        _, ayp, byp = xyY_to_lab(x0, y0 + h, Y0)
        _, aym, bym = xyY_to_lab(x0, y0 - h, Y0)
        Jab = np.array([
            [(axp - axm) / (2 * h), (ayp - aym) / (2 * h)],
            [(bxp - bxm) / (2 * h), (byp - bym) / (2 * h)],
        ])

        # Transform observed metric to (x, y)
        G_obs_xy = Jab.T @ G_ab @ Jab
        if np.linalg.eigvalsh(G_obs_xy)[0] <= 0:
            continue

        for ang in angles:
            v = np.array([np.cos(ang), np.sin(ang)])
            so = v @ G_obs_xy @ v
            sp = v @ g_pred @ v
            if so > 0 and sp > 0:
                obs.append(1.0 / np.sqrt(so))
                pred.append(1.0 / np.sqrt(sp))

    if len(obs) < 10:
        return 999.0
    return stress(obs, pred)


# =====================================================================
#  Diagnostic measures
# =====================================================================

def orientation_errors(data, metric_fn):
    """Compute orientation error (degrees) for each MacAdam ellipse."""
    errors = []
    for ell in data:
        gp = metric_fn(ell["x"], ell["y"], ell["Y"])
        if gp is None:
            continue
        # Observed major axis direction
        _, P_obs = np.linalg.eigh(np.linalg.inv(ell["g_obs"]))
        t_obs = np.degrees(np.arctan2(P_obs[1, -1], P_obs[0, -1])) % 180
        # Predicted major axis direction
        _, P_pred = np.linalg.eigh(np.linalg.inv(gp))
        t_pred = np.degrees(np.arctan2(P_pred[1, -1], P_pred[0, -1])) % 180
        err = min(abs(t_pred - t_obs), 180 - abs(t_pred - t_obs))
        errors.append(err)
    return errors


def frobenius_mismatch(S_pred, S_obs):
    """Relative Frobenius mismatch after optimal scaling."""
    ss = np.sum(S_pred * S_pred)
    if ss < 1e-30:
        return 1.0
    alpha = np.sum(S_pred * S_obs) / ss
    Sa = alpha * S_pred
    d = np.linalg.norm(Sa - S_obs, "fro")
    t = np.linalg.norm(Sa, "fro") + np.linalg.norm(S_obs, "fro")
    if t < 1e-30:
        return 0.0
    return d / t


def frobenius_quartiles(data, sigma_fn):
    """Compute Q1, Q2, Q3 of Frobenius mismatch across Koenderink data."""
    vals = []
    for ell in data:
        Sp = sigma_fn(ell)
        if Sp is not None:
            vals.append(frobenius_mismatch(Sp, ell["Sigma"]))
    vals.sort()
    n = len(vals)
    if n == 0:
        return 999.0, 999.0, 999.0
    return vals[n // 4], vals[n // 2], vals[3 * n // 4]


# =====================================================================
#  Main
# =====================================================================

def main():
    print("=" * 65)
    print("Replication of paper results")
    print("17-parameter biological model (Model S)")
    print("=" * 65)

    mac_data = load_macadam()
    koen_data = load_koenderink()

    # -----------------------------------------------------------------
    #  Evaluate the biological model on all four datasets
    # -----------------------------------------------------------------

    def bio_mac(x, y, Y):
        return bio_metric_2d(x, y, Y, PARAMS, LMS_ILLC)

    def bio_koen(entry):
        return bio_sigma_3d(entry, PARAMS)

    def bio_huang(x, y, Y, adapt):
        return bio_metric_2d(x, y, Y, PARAMS, adapt)

    s_mac    = stress_2d(mac_data, bio_mac)
    s_koen   = stress_3d(koen_data, bio_koen)
    s_wright = stress_wright(WRIGHT, bio_mac)
    s_huang  = stress_huang(HUANG, bio_huang, LMS_D65)
    t4 = s_mac + s_koen + s_wright + s_huang

    print(f"\nBiological model (17p):")
    print(f"  MacAdam:     {s_mac:.1f}")
    print(f"  Koenderink:  {s_koen:.1f}")
    print(f"  Wright:      {s_wright:.1f}")
    print(f"  Huang:       {s_huang:.1f}")
    print(f"  Total T4:    {t4:.1f}")

    oe = orientation_errors(mac_data, bio_mac)
    fq = frobenius_quartiles(koen_data, bio_koen)
    print(f"\n  MacAdam orientation error:  median = {np.median(oe):.1f} deg, "
          f"mean = {np.mean(oe):.1f} deg")
    print(f"  Koenderink Frobenius:  Q1 = {fq[0]:.3f}, "
          f"Q2 = {fq[1]:.3f}, Q3 = {fq[2]:.3f}")

    # -----------------------------------------------------------------
    #  CIELAB baseline
    # -----------------------------------------------------------------

    s_mac_lab = stress_2d(mac_data, cielab_metric_2d)
    s_wright_lab = stress_wright(WRIGHT, cielab_metric_2d)
    print(f"\nCIELAB:")
    print(f"  MacAdam:  {s_mac_lab:.1f}    Wright:  {s_wright_lab:.1f}")

    # -----------------------------------------------------------------
    #  Optional: colour-science metrics
    # -----------------------------------------------------------------

    colour = _try_import_colour()
    if colour is not None:
        cam = make_cam_metrics(colour)

        def wrap(mf):
            return lambda x, y, Y, adapt: mf(x, y, Y)

        for name in ["CIEDE2000", "CIECAM02-UCS", "CAM16-UCS"]:
            mf = cam[name]
            sm = stress_2d(mac_data, mf)
            sw = stress_wright(WRIGHT, mf)
            sh = stress_huang(HUANG, wrap(mf), LMS_D65)
            print(f"\n{name}:")
            print(f"  MacAdam:  {sm:.1f}    Wright:  {sw:.1f}    Huang:  {sh:.1f}")
    else:
        print("\n(colour-science not installed; skipping CIEDE2000 / CAM metrics)")

    # -----------------------------------------------------------------
    #  Summary table
    # -----------------------------------------------------------------

    print(f"\n{'=' * 65}")
    print(f"{'Metric':<20} {'MacAdam':>8} {'Koenderink':>11} "
          f"{'Wright':>8} {'Huang':>8}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 11} {'-' * 8} {'-' * 8}")
    print(f"{'CIELAB':<20} {s_mac_lab:>8.1f} {'--':>11} "
          f"{s_wright_lab:>8.1f} {'--':>8}")
    if colour is not None:
        for name in ["CIEDE2000", "CIECAM02-UCS", "CAM16-UCS"]:
            mf = cam[name]
            sm = stress_2d(mac_data, mf)
            sw = stress_wright(WRIGHT, mf)
            sh = stress_huang(HUANG, wrap(mf), LMS_D65)
            print(f"{name:<20} {sm:>8.1f} {'--':>11} {sw:>8.1f} {sh:>8.1f}")
    print(f"{'Biological (17p)':<20} {s_mac:>8.1f} {s_koen:>11.1f} "
          f"{s_wright:>8.1f} {s_huang:>8.1f}")
    print(f"{'=' * 65}")

    # -----------------------------------------------------------------
    #  Huang per-centre breakdown
    # -----------------------------------------------------------------

    print(f"\nHuang per-centre STRESS:")
    for i, (L, al, bl, A, AB, th) in enumerate(HUANG):
        single = [(L, al, bl, A, AB, th)]
        si = stress_huang(single, bio_huang, LMS_D65)
        xi, yi, Yi = lab_to_xyY(L, al, bl)
        print(f"  {HUANG_NAMES[i]:<16} {si:5.1f}   "
              f"(x={xi:.3f}, y={yi:.3f}, Y={Yi:.1f})")

    # -----------------------------------------------------------------
    #  Wright per-wavelength analysis
    # -----------------------------------------------------------------

    w0 = 0.5 ** PARAMS[0] / (0.5 ** PARAMS[0] + PARAMS[1])
    print(f"\nWright per-wavelength (w0 = {w0:.5f}):")
    print(f"  {'lambda':>6} {'obs':>5} {'w':>8} {'w/w0':>7} {'S(w)':>7}")
    for lam, dlam, x, y, dxdl, dydl in WRIGHT:
        zw = opponent_coords(x, y, 48.0, PARAMS, LMS_ILLC)
        if zw is None:
            continue
        w = zw[1]
        S_val = 1 + PARAMS[15] * PARAMS[16] ** 2 / (w ** 2 + PARAMS[16] ** 2)
        print(f"  {lam:>6.0f} {dlam:>5.1f} {w:>8.5f} {w / w0:>7.3f} {S_val:>7.2f}")

    # -----------------------------------------------------------------
    #  Ablation studies
    # -----------------------------------------------------------------

    print(f"\n{'=' * 65}")
    print("Ablation studies")
    print(f"{'=' * 65}")

    # 1. eta = 0 (remove non-cardinal population entirely)
    p_no_eta = PARAMS.copy()
    p_no_eta[3] = 0.0

    def abl_eta_mac(x, y, Y):
        return bio_metric_2d(x, y, Y, p_no_eta, LMS_ILLC)
    def abl_eta_koen(entry):
        return bio_sigma_3d(entry, p_no_eta)
    def abl_eta_huang(x, y, Y, adapt):
        return bio_metric_2d(x, y, Y, p_no_eta, adapt)

    ae_mac = stress_2d(mac_data, abl_eta_mac)
    ae_koen = stress_3d(koen_data, abl_eta_koen)
    ae_wri = stress_wright(WRIGHT, abl_eta_mac)
    ae_hua = stress_huang(HUANG, abl_eta_huang, LMS_D65)

    print(f"\neta = 0 (no non-cardinal population):")
    print(f"  MacAdam:    {s_mac:.1f} -> {ae_mac:.1f}  ({ae_mac - s_mac:+.1f})")
    print(f"  Koenderink: {s_koen:.1f} -> {ae_koen:.1f}  ({ae_koen - s_koen:+.1f})")
    print(f"  Wright:     {s_wright:.1f} -> {ae_wri:.1f}  ({ae_wri - s_wright:+.1f})")
    print(f"  Huang:      {s_huang:.1f} -> {ae_hua:.1f}  ({ae_hua - s_huang:+.1f})")

    # 2. delta_nc = 0 (disable S-cone disinhibition, revert to constant eta)
    p_no_disinh = PARAMS.copy()
    p_no_disinh[15] = 0.0

    def abl_disinh_mac(x, y, Y):
        return bio_metric_2d(x, y, Y, p_no_disinh, LMS_ILLC)
    def abl_disinh_koen(entry):
        return bio_sigma_3d(entry, p_no_disinh)
    def abl_disinh_huang(x, y, Y, adapt):
        return bio_metric_2d(x, y, Y, p_no_disinh, adapt)

    ad_mac = stress_2d(mac_data, abl_disinh_mac)
    ad_koen = stress_3d(koen_data, abl_disinh_koen)
    ad_wri = stress_wright(WRIGHT, abl_disinh_mac)
    ad_hua = stress_huang(HUANG, abl_disinh_huang, LMS_D65)

    print(f"\ndelta_nc = 0 (no S-cone disinhibition):")
    print(f"  MacAdam:    {s_mac:.1f} -> {ad_mac:.1f}  ({ad_mac - s_mac:+.1f})")
    print(f"  Koenderink: {s_koen:.1f} -> {ad_koen:.1f}  ({ad_koen - s_koen:+.1f})")
    print(f"  Wright:     {s_wright:.1f} -> {ad_wri:.1f}  ({ad_wri - s_wright:+.1f})")
    print(f"  Huang:      {s_huang:.1f} -> {ad_hua:.1f}  ({ad_hua - s_huang:+.1f})")

    # 3. tau -> infinity (disable konio cross-normalization)
    p_no_tau = PARAMS.copy()
    p_no_tau[14] = 1e6

    def abl_tau_mac(x, y, Y):
        return bio_metric_2d(x, y, Y, p_no_tau, LMS_ILLC)
    def abl_tau_koen(entry):
        return bio_sigma_3d(entry, p_no_tau)
    def abl_tau_huang(x, y, Y, adapt):
        return bio_metric_2d(x, y, Y, p_no_tau, adapt)

    at_mac = stress_2d(mac_data, abl_tau_mac)
    at_koen = stress_3d(koen_data, abl_tau_koen)
    at_wri = stress_wright(WRIGHT, abl_tau_mac)
    at_hua = stress_huang(HUANG, abl_tau_huang, LMS_D65)

    print(f"\ntau -> inf (no konio cross-normalization):")
    print(f"  MacAdam:    {s_mac:.1f} -> {at_mac:.1f}  ({at_mac - s_mac:+.1f})")
    print(f"  Koenderink: {s_koen:.1f} -> {at_koen:.1f}  ({at_koen - s_koen:+.1f})")
    print(f"  Wright:     {s_wright:.1f} -> {at_wri:.1f}  ({at_wri - s_wright:+.1f})")
    print(f"  Huang:      {s_huang:.1f} -> {at_hua:.1f}  ({at_hua - s_huang:+.1f})")

    # -----------------------------------------------------------------
    #  Parameter summary
    # -----------------------------------------------------------------

    print(f"\n{'=' * 65}")
    print("Fitted parameters")
    print(f"{'=' * 65}")
    for i, (name, val) in enumerate(zip(PARAM_NAMES, PARAMS)):
        print(f"  {i:>2}  {name:<12}  {val:.6f}")


if __name__ == "__main__":
    main()
