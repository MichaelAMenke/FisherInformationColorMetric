#!/usr/bin/env python3
"""Generate paper figures for Model S (17p constrained)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb

# =====================================================================
#  Model S parameters (constrained weighted-polish result)
# =====================================================================

PARAMS_S = np.array([
    0.596802,    # p_s
    8.616093,    # kappa_s
    12.854079,   # gamma
    2.354558,    # eta
    -2.243597,   # theta_0
    0.993793,    # theta_1
    1.508492,    # theta_2
    1.993487,    # z_max
    1.021526,    # p_z
    1.392899,    # kappa_z
    0.716583,    # a_P
    0.567920,    # a_K
    0.737806,    # psi_0
    0.840674,    # beta_Y
    0.796076,    # tau
    10.000691,   # delta_nc
    0.011876,    # sigma_w
])

# =====================================================================
#  Constants
# =====================================================================

M_SP = np.array([
    [ 0.15514,  0.54312, -0.03286],
    [-0.15514,  0.45684,  0.03286],
    [ 0.0,      0.0,      0.01608],
])

_M_sRGB_UNIT = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
Y_WHITE = 214.0
M_sRGB = _M_sRGB_UNIT * Y_WHITE
_H = 1e-7
XN_10, YN_10, ZN_10 = 94.811, 100.0, 107.304
Y_REF = 48.0

# =====================================================================
#  Datasets
# =====================================================================

MACADAM = [
    (0.160,0.057,0.85,0.35,62.5),(0.187,0.118,2.20,0.55,77.0),
    (0.253,0.125,2.50,0.50,55.5),(0.150,0.680,9.60,2.30,105.0),
    (0.131,0.521,4.70,2.00,112.5),(0.212,0.550,5.80,2.30,100.0),
    (0.258,0.450,5.00,2.00,92.0),(0.152,0.365,3.80,1.90,110.0),
    (0.280,0.385,4.00,1.50,75.5),(0.380,0.498,4.40,1.20,70.0),
    (0.160,0.200,2.10,0.95,104.0),(0.228,0.250,3.10,0.90,72.0),
    (0.305,0.323,2.30,0.90,58.0),(0.385,0.393,3.80,1.60,65.5),
    (0.472,0.399,3.20,1.40,51.0),(0.527,0.350,2.60,1.30,20.0),
    (0.475,0.300,2.90,1.10,28.5),(0.510,0.236,2.40,1.20,29.5),
    (0.596,0.283,2.60,1.30,13.0),(0.344,0.284,2.30,0.90,60.0),
    (0.390,0.237,2.50,1.00,47.0),(0.441,0.198,2.80,0.95,34.5),
    (0.278,0.223,2.40,0.55,57.5),(0.300,0.163,2.90,0.60,54.0),
    (0.365,0.153,3.60,0.95,40.0),
]

KOENDERINK = [
    (200,200,200,5921,1948,928,3886,1021,4749),
    (200,200,500,9215,1360,4103,11861,1386,15525),
    (200,200,800,21690,-3347,10627,48696,-3215,36789),
    (200,500,200,35342,1900,2635,14007,3373,27777),
    (200,500,500,24694,3463,1482,12425,5167,14643),
    (200,500,800,26625,1302,5918,17913,3172,32615),
    (200,800,200,165975,5955,2116,28389,6863,41825),
    (200,800,500,73111,9497,5039,28421,9518,23966),
    (200,800,800,101190,-2606,6015,16789,6767,20554),
    (350,350,350,6783,3291,2856,8216,3095,10971),
    (350,350,650,9701,3105,749,12794,742,22673),
    (350,650,350,25572,6261,8090,18852,5571,25738),
    (350,650,650,23178,6578,6647,21049,8966,17971),
    (500,200,200,14834,2552,2079,10992,4923,12283),
    (500,200,500,15568,3162,4692,20208,3389,20117),
    (500,200,800,14308,3288,7870,22941,-489,20016),
    (500,500,200,15405,6395,1161,13440,4827,23050),
    (500,500,500,9587,5295,4776,8469,6427,14371),
    (500,500,800,15252,4851,7727,13422,2622,25493),
    (500,800,200,37494,12352,8210,32347,7589,39717),
    (500,800,500,19748,5502,2284,22837,8286,19868),
    (500,800,800,22456,7329,6247,21250,9843,21454),
    (650,350,350,15701,4312,2756,12205,5799,11475),
    (650,350,650,16335,5107,6724,19765,157,26332),
    (650,650,350,19619,8583,7085,15874,5627,18256),
    (650,650,650,17727,13937,11284,17931,12472,19410),
    (800,200,200,28293,4685,5244,23305,6907,16204),
    (800,200,500,22889,6518,12858,35380,6847,26719),
    (800,200,800,26613,4662,9539,64125,-556,44888),
    (800,500,200,19053,6333,6568,16388,5754,20538),
    (800,500,500,22663,9771,8971,17408,9786,20296),
    (800,500,800,21846,6681,11800,20108,6351,31972),
    (800,800,200,23604,13996,500,21787,-739,32105),
    (800,800,500,19239,8751,9896,19582,11909,23151),
    (800,800,800,19382,6859,7526,14190,7048,21049),
]

HUANG = [
    (63.2,-0.3,0.3,1.25,1.74,64.53),(46.6,37.8,23.0,1.92,1.98,60.89),
    (44.8,58.2,36.9,2.27,2.05,46.18),(64.4,13.5,22.4,2.13,3.01,82.29),
    (61.6,34.2,63.5,3.71,4.49,86.98),(86.5,-6.3,48.0,1.91,1.94,92.42),
    (86.2,-9.5,72.7,2.79,4.16,85.90),(66.5,-10.6,14.0,1.25,1.63,99.53),
    (65.9,-30.4,41.4,3.02,4.19,111.58),(57.9,-32.5,1.0,1.23,1.30,142.72),
    (58.7,-37.8,0.6,1.78,2.15,161.37),(51.4,-16.7,-10.3,1.10,1.54,3.57),
    (52.6,-27.1,-17.8,1.31,1.61,61.14),(37.9,5.5,-31.8,1.05,1.49,122.00),
    (36.7,6.2,-44.4,1.22,1.54,131.51),(46.9,11.5,-13.3,2.15,3.40,105.21),
    (46.7,26.3,-26.3,1.96,3.05,128.78),
]

# =====================================================================
#  Core functions
# =====================================================================

def xyY_to_XYZ(x,y,Y):
    if y<1e-10: return np.array([0.,Y,0.])
    return np.array([Y*x/y,Y,Y*(1-x-y)/y])

def xyY_to_LMS(x,y,Y):
    return np.maximum(M_SP @ xyY_to_XYZ(x,y,Y), 1e-12)

def srgb_gamma_decode(v):
    return v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4

def rgb_to_xyY(rgb):
    rl = np.array([srgb_gamma_decode(c) for c in rgb])
    XYZ = M_sRGB @ rl; s = sum(XYZ)
    if s<1e-10: return np.array([0.3127,0.3290,1e-6])
    return np.array([XYZ[0]/s, XYZ[1]/s, XYZ[1]])

LMS_ILLC = xyY_to_LMS(0.31006, 0.31616, Y_REF)
LMS_D65  = xyY_to_LMS(0.31272, 0.32903, Y_REF)

def opponent_coords(x,y,Y,params,adapt_lms):
    p_s,kappa_s = params[0],params[1]
    z_max,p_z,kappa_z = params[7],params[8],params[9]
    lms = xyY_to_LMS(x,y,Y); la = lms/adapt_lms
    if np.any(la<1e-15): return None
    u = np.log(la[0]/la[1]); au = abs(u)
    z = 0. if au<1e-15 else z_max*np.sign(u)*au**p_z/(au**p_z+kappa_z)
    r = la[2]/(la[0]+la[1])
    if r<1e-15: return None
    rp = r**p_s; w = rp/(rp+kappa_s)
    return np.array([z,w])

def bio_metric_2d(x,y,Y,params,adapt_lms):
    p_s,kappa_s = params[0],params[1]
    gamma,eta = params[2],params[3]
    theta0,theta1,theta2 = params[4],params[5],params[6]
    a_P,a_K = params[10],params[11]
    tau = params[14]
    delta_nc,sigma_w = params[15],params[16]
    w0 = 0.5**p_s/(0.5**p_s+kappa_s)

    def oc(cx,cy): return opponent_coords(cx,cy,Y,params,adapt_lms)
    zw = oc(x,y)
    if zw is None: return None
    a,b = oc(x+_H,y),oc(x-_H,y)
    c,d = oc(x,y+_H),oc(x,y-_H)
    if any(v is None for v in [a,b,c,d]): return None
    J = np.column_stack([(a-b)/(2*_H),(c-d)/(2*_H)])
    z,w = zw[0],zw[1]; dw = w-w0
    theta = theta0+theta1*z+theta2*dw
    ndir = np.array([np.cos(theta),np.sin(theta)])
    N_K = tau/(tau+z*z)
    nc_boost = 1.0+delta_nc*sigma_w**2/(w**2+sigma_w**2)
    Ys = max(Y,0.01)
    G = np.diag([Ys**a_P, N_K*gamma**2*Ys**a_K])
    G += eta**2*nc_boost*Ys**((a_P+a_K)/2)*np.outer(ndir,ndir)
    g = J.T@G@J
    if np.linalg.eigvalsh(g)[0]<=0: return None
    return g

def bio_sigma_3d(entry, params, adapt_lms):
    psi_0,beta_Y = params[12],params[13]
    rgb = entry["rgb"]; xyY = rgb_to_xyY(rgb)
    x0,y0,Y0 = xyY
    if Y0<1e-6: return None
    gc = bio_metric_2d(x0,y0,Y0,params,adapt_lms)
    if gc is None: return None
    Ys = max(Y0,0.01)
    g3 = np.zeros((3,3)); g3[:2,:2] = gc; g3[2,2] = psi_0**2/Ys**(2*beta_Y)
    Jf = np.zeros((3,3))
    for j in range(3):
        rp,rm = rgb.copy(),rgb.copy()
        rp[j] = min(rp[j]+_H,1.); rm[j] = max(rm[j]-_H,0.)
        delta = rp[j]-rm[j]
        if delta<1e-12: return None
        Jf[:,j] = (rgb_to_xyY(rp)-rgb_to_xyY(rm))/delta
    if abs(np.linalg.det(Jf))<1e-20: return None
    gr = Jf.T@g3@Jf
    if np.linalg.eigvalsh(gr)[0]<=0: return None
    return np.linalg.inv(gr)

# =====================================================================
#  STRESS with optimal F
# =====================================================================

def compute_stress_and_F(observed, predicted):
    o,p = np.asarray(observed),np.asarray(predicted)
    F = np.dot(o,p)/np.dot(p,p)
    S = 100*np.sqrt(np.sum((o-F*p)**2)/np.sum(o**2))
    return S, F

def get_2d_F(data, params, adapt_lms, n_angles=12):
    angles = np.linspace(0,np.pi,n_angles,endpoint=False)
    obs,pred = [],[]
    for ell in data:
        g = bio_metric_2d(ell["x"],ell["y"],ell["Y"],params,adapt_lms)
        if g is None: continue
        for a in angles:
            v = np.array([np.cos(a),np.sin(a)])
            ro = v@ell["g_obs"]@v; rp = v@g@v
            if ro>0 and rp>0:
                obs.append(1/np.sqrt(ro)); pred.append(1/np.sqrt(rp))
    return compute_stress_and_F(obs,pred)

# =====================================================================
#  Ellipse from metric tensor
# =====================================================================

def metric_to_ellipse_params(g):
    """Returns (a, b, theta_deg) where a>=b are semi-axes of g^{-1}."""
    Sig = np.linalg.inv(g)
    eigvals, eigvecs = np.linalg.eigh(Sig)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:,idx]
    a = np.sqrt(eigvals[0]); b = np.sqrt(eigvals[1])
    theta = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    return a, b, theta

# =====================================================================
#  CIELAB conversions for Huang
# =====================================================================

def lab_to_xyY(L,a,b):
    fy=(L+16)/116; fx=fy+a/500; fz=fy-b/200; d=6/29
    def fi(t): return t**3 if t>d else 3*d**2*(t-4/29)
    X=XN_10*fi(fx); Y=YN_10*fi(fy); Z=ZN_10*fi(fz); s=X+Y+Z
    if s<1e-10: return 0.3127,0.329,0.01
    return X/s,Y/s,Y

def xyY_to_lab(x,y,Yv):
    if y<1e-10: return 0.,0.,0.
    X=Yv*x/y; Z=Yv*(1-x-y)/y; d3=(6/29)**3
    def fw(t): return t**(1/3) if t>d3 else t/(3*(6/29)**2)+4/29
    return 116*fw(Yv/YN_10)-16, 500*(fw(X/XN_10)-fw(Yv/YN_10)), 200*(fw(Yv/YN_10)-fw(Z/ZN_10))

# =====================================================================
#  Spectrum locus for plotting
# =====================================================================

# CIE 1931 2-deg chromaticity coordinates at selected wavelengths
SPECTRUM_LOCUS = [
    (380,0.1741,0.0050),(390,0.1740,0.0050),(400,0.1734,0.0049),
    (410,0.1714,0.0051),(420,0.1689,0.0069),(430,0.1644,0.0109),
    (440,0.1566,0.0177),(450,0.1440,0.0297),(460,0.1241,0.0578),
    (470,0.0913,0.1327),(480,0.0454,0.2950),(490,0.0082,0.5384),
    (500,0.0139,0.7502),(510,0.0743,0.8338),(520,0.1547,0.8059),
    (530,0.2296,0.7543),(540,0.3016,0.6923),(550,0.3731,0.6245),
    (560,0.4441,0.5547),(570,0.5125,0.4866),(580,0.5752,0.4242),
    (590,0.6270,0.3725),(600,0.6658,0.3340),(610,0.6915,0.3083),
    (620,0.7006,0.2993),(630,0.7079,0.2920),(640,0.7140,0.2859),
    (650,0.7190,0.2809),(660,0.7230,0.2770),(670,0.7260,0.2740),
    (680,0.7283,0.2717),(690,0.7300,0.2700),(700,0.7347,0.2653),
]

# =====================================================================
#  Figure 1: MacAdam ellipses in CIE (x,y)
# =====================================================================

def plot_macadam():
    print("Generating MacAdam figure...", flush=True)
    
    # Load data
    data = []
    for x,y,a3,b3,th in MACADAM:
        t = np.radians(th); c,s = np.cos(t),np.sin(t)
        R = np.array([[c,-s],[s,c]])
        g = R@np.diag([1/(a3*1e-3)**2, 1/(b3*1e-3)**2])@R.T
        data.append({"x":x,"y":y,"Y":Y_REF,"g_obs":g,
                      "a":a3*1e-3,"b":b3*1e-3,"theta":th})

    # Get optimal scale factor
    stress_val, F = get_2d_F(data, PARAMS_S, LMS_ILLC)
    print(f"  MacAdam STRESS = {stress_val:.1f}, F = {F:.4f}")
    mag = 10  # magnification

    fig, ax = plt.subplots(1,1,figsize=(8,7))

    # Spectrum locus
    sl_x = [p[1] for p in SPECTRUM_LOCUS]
    sl_y = [p[2] for p in SPECTRUM_LOCUS]
    ax.plot(sl_x, sl_y, 'k-', lw=0.8, alpha=0.4)
    ax.plot([sl_x[0],sl_x[-1]], [sl_y[0],sl_y[-1]], 'k-', lw=0.8, alpha=0.4)

    for ell in data:
        xc, yc = ell["x"], ell["y"]

        # Observed ellipse
        a_obs = ell["a"]*mag; b_obs = ell["b"]*mag; th_obs = ell["theta"]
        e_obs = Ellipse((xc,yc), 2*a_obs, 2*b_obs, angle=th_obs,
                        fill=False, edgecolor='#2166ac', lw=1.2)
        ax.add_patch(e_obs)

        # Predicted ellipse
        g_pred = bio_metric_2d(xc,yc,ell["Y"],PARAMS_S,LMS_ILLC)
        if g_pred is not None:
            # Scale by F: predicted radius -> F*predicted radius
            # g_scaled = g_pred / F^2
            g_scaled = g_pred / (F**2)
            a_p, b_p, th_p = metric_to_ellipse_params(g_scaled)
            e_pred = Ellipse((xc,yc), 2*a_p*mag, 2*b_p*mag, angle=th_p,
                            fill=False, edgecolor='#b2182b', lw=1.2,
                            linestyle='--')
            ax.add_patch(e_pred)

    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0.0, 0.90)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title(f'MacAdam (1942) — observed (blue) vs predicted (red dashed), '
                 f'$10\\times$ mag.  STRESS = {stress_val:.1f}', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/fig_macadam.pdf', dpi=150)
    fig.savefig('/mnt/user-data/outputs/fig_macadam.png', dpi=150)
    plt.close(fig)
    print(f"  Saved fig_macadam.pdf/png")

# =====================================================================
#  Figure 2: Huang ellipses in CIE (x,y)
# =====================================================================

def plot_huang():
    print("Generating Huang figure...", flush=True)

    h_step = 1e-5
    n_angles = 12
    mag = 5

    # Build data with observed metrics in (x,y)
    data = []
    for (L,a_lab,b_lab,A,AB_ratio,theta_deg) in HUANG:
        x0,y0,Y0 = lab_to_xyY(L,a_lab,b_lab)
        if y0<1e-6 or Y0<0.1: continue
        B_val = A/AB_ratio
        theta = theta_deg*np.pi/180
        ct,st = np.cos(theta),np.sin(theta)
        iA,iB = 1/(A*A),1/(B_val*B_val)
        G_ab = np.array([[ct*ct*iA+st*st*iB, ct*st*(iA-iB)],
                         [ct*st*(iA-iB), st*st*iA+ct*ct*iB]])
        _,ap,bp = xyY_to_lab(x0+h_step,y0,Y0)
        _,am,bm = xyY_to_lab(x0-h_step,y0,Y0)
        _,apy,bpy = xyY_to_lab(x0,y0+h_step,Y0)
        _,amy,bmy = xyY_to_lab(x0,y0-h_step,Y0)
        Jab = np.array([[(ap-am)/(2*h_step),(apy-amy)/(2*h_step)],
                        [(bp-bm)/(2*h_step),(bpy-bmy)/(2*h_step)]])
        G_obs_xy = Jab.T@G_ab@Jab
        if np.linalg.eigvalsh(G_obs_xy)[0]<=0: continue
        data.append({"x":x0,"y":y0,"Y":Y0,"g_obs":G_obs_xy})

    # Compute F
    stress_val, F = get_2d_F(data, PARAMS_S, LMS_D65)
    print(f"  Huang STRESS = {stress_val:.1f}, F = {F:.4f}")

    fig, ax = plt.subplots(1,1,figsize=(8,7))

    sl_x = [p[1] for p in SPECTRUM_LOCUS]
    sl_y = [p[2] for p in SPECTRUM_LOCUS]
    ax.plot(sl_x, sl_y, 'k-', lw=0.8, alpha=0.4)
    ax.plot([sl_x[0],sl_x[-1]], [sl_y[0],sl_y[-1]], 'k-', lw=0.8, alpha=0.4)

    for ell in data:
        xc, yc = ell["x"], ell["y"]

        # Observed
        a_o, b_o, th_o = metric_to_ellipse_params(ell["g_obs"])
        e_obs = Ellipse((xc,yc), 2*a_o*mag, 2*b_o*mag, angle=th_o,
                        fill=False, edgecolor='#2166ac', lw=1.2)
        ax.add_patch(e_obs)

        # Predicted
        g_pred = bio_metric_2d(xc,yc,ell["Y"],PARAMS_S,LMS_D65)
        if g_pred is not None:
            g_scaled = g_pred / (F**2)
            a_p, b_p, th_p = metric_to_ellipse_params(g_scaled)
            e_pred = Ellipse((xc,yc), 2*a_p*mag, 2*b_p*mag, angle=th_p,
                            fill=False, edgecolor='#b2182b', lw=1.2,
                            linestyle='--')
            ax.add_patch(e_pred)

    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0.0, 0.90)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title(f'Huang et al. (2012) — observed (blue) vs predicted (red dashed), '
                 f'$5\\times$ mag.  STRESS = {stress_val:.1f}', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/fig_huang.pdf', dpi=150)
    fig.savefig('/mnt/user-data/outputs/fig_huang.png', dpi=150)
    plt.close(fig)
    print(f"  Saved fig_huang.pdf/png")

# =====================================================================
#  Figure 3: Koenderink ellipsoids — 3 projected panels (RG, RB, GB)
# =====================================================================

def plot_koenderink():
    print("Generating Koenderink figure...", flush=True)

    # Load data
    koen_data = []
    for row in KOENDERINK:
        rgb = np.array(row[:3],dtype=float)/1000.
        s11,s12,s13,s22,s23,s33 = [v/1e7 for v in row[3:]]
        Sigma = np.array([[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]])
        xyY = rgb_to_xyY(rgb)
        koen_data.append({"rgb":rgb.copy(),"Sigma":Sigma,
                          "x":xyY[0],"y":xyY[1],"Y":xyY[2]})

    # Compute global F for 3D
    def fib_dirs(n=26):
        g=(1+np.sqrt(5))/2; d=[]
        for i in range(n):
            th=np.arccos(1-2*(i+0.5)/n); ph=2*np.pi*i/g
            d.append(np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]))
        return d
    dirs = fib_dirs()
    obs_all, pred_all = [], []
    for ell in koen_data:
        Sp = bio_sigma_3d(ell, PARAMS_S, LMS_D65)
        if Sp is None: continue
        for d in dirs:
            ro = np.sqrt(d@ell["Sigma"]@d)
            rp = np.sqrt(d@Sp@d)
            if ro>0 and rp>0:
                obs_all.append(ro); pred_all.append(rp)
    stress_val, F_3d = compute_stress_and_F(obs_all, pred_all)
    print(f"  Koenderink STRESS = {stress_val:.1f}, F = {F_3d:.4f}")

    mag = 2.5
    pairs = [(0,1,'R','G'), (0,2,'R','B'), (1,2,'G','B')]
    fig, axes = plt.subplots(1,3,figsize=(16,5.5))

    for pidx, (i,j,lab_i,lab_j) in enumerate(pairs):
        ax = axes[pidx]

        # Greedy subset selection: pick ellipsoids well-separated in
        # this 2D projection to avoid overlap
        centers = [(ell["rgb"][i], ell["rgb"][j]) for ell in koen_data]
        selected = []
        min_dist = 0.22  # minimum distance between centres in this projection
        for kidx, (ci, cj) in enumerate(centers):
            too_close = False
            for sidx in selected:
                si, sj = centers[sidx]
                if np.sqrt((ci-si)**2 + (cj-sj)**2) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append(kidx)

        n_shown = 0
        for kidx in selected:
            ell = koen_data[kidx]
            n_shown += 1
            rgb = ell["rgb"]
            c_i, c_j = rgb[i], rgb[j]

            # Observed 2D projection
            Sig = ell["Sigma"]
            Sig_2d = np.array([[Sig[i,i],Sig[i,j]],[Sig[i,j],Sig[j,j]]])
            eigvals_o, eigvecs_o = np.linalg.eigh(Sig_2d)
            if np.any(eigvals_o<=0): continue
            idx_sort = np.argsort(eigvals_o)[::-1]
            a_o = np.sqrt(eigvals_o[idx_sort[0]])*mag
            b_o = np.sqrt(eigvals_o[idx_sort[1]])*mag
            th_o = np.degrees(np.arctan2(eigvecs_o[1,idx_sort[0]],
                                         eigvecs_o[0,idx_sort[0]]))
            e_obs = Ellipse((c_i,c_j), 2*a_o, 2*b_o, angle=th_o,
                           fill=False, edgecolor='#2166ac', lw=1.0)
            ax.add_patch(e_obs)

            # Predicted
            Sp = bio_sigma_3d(ell, PARAMS_S, LMS_D65)
            if Sp is not None:
                Sp_scaled = Sp * F_3d**2
                Sp_2d = np.array([[Sp_scaled[i,i],Sp_scaled[i,j]],
                                  [Sp_scaled[i,j],Sp_scaled[j,j]]])
                eigvals_p, eigvecs_p = np.linalg.eigh(Sp_2d)
                if np.all(eigvals_p>0):
                    idx_sort_p = np.argsort(eigvals_p)[::-1]
                    a_p = np.sqrt(eigvals_p[idx_sort_p[0]])*mag
                    b_p = np.sqrt(eigvals_p[idx_sort_p[1]])*mag
                    th_p = np.degrees(np.arctan2(eigvecs_p[1,idx_sort_p[0]],
                                                 eigvecs_p[0,idx_sort_p[0]]))
                    e_pred = Ellipse((c_i,c_j), 2*a_p, 2*b_p, angle=th_p,
                                   fill=False, edgecolor='#b2182b', lw=1.0,
                                   linestyle='--')
                    ax.add_patch(e_pred)

            # Dot colored by sRGB
            clr = np.clip(rgb, 0, 1)
            ax.plot(c_i, c_j, 'o', color=clr, markersize=3, zorder=5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f'${lab_i}_\\gamma$', fontsize=12)
        ax.set_ylabel(f'${lab_j}_\\gamma$', fontsize=12)
        ax.set_title(f'{lab_i}-{lab_j} projection ({n_shown} ellipsoids)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Koenderink et al. (2026) — observed (blue) vs predicted '
                 f'(red dashed), $2.5\\times$ mag.  STRESS = {stress_val:.1f}',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/fig_koenderink.pdf', dpi=150,
                bbox_inches='tight')
    fig.savefig('/mnt/user-data/outputs/fig_koenderink.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig_koenderink.pdf/png")

# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating paper figures for Model S (17p constrained)")
    print("=" * 60)
    plot_macadam()
    plot_huang()
    plot_koenderink()
    print("\nDone.")
