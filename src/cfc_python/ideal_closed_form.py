
# ideal_closed_form.py
# Closed form replacement for cfc_core's original dblquad integral, ensures that cfc_core remains valid when conf_noise is small, including the conf_noise -> 0 limit. This occurs for all ideal/super-ideal reference computations, causing numerical issues and instability in the fits
 
# U1 = (x1-mu1)/sens_noise1
# U2 = (x2-mu2)/sens_noise2
# Both are ~ iid N(0,1)
 
# C=1 iff  C1 - C2 + (confidence noise) > -intrvl_bias
# C_i = interval i's decision-signed confidence evidence, which can be expressed as an affine function of U_i.

# The two confidence-noise terms are independent Normals, so together they are one further independent Normal Z with a combined stdev S = sqrt(conf_noise1^2 + conf_noise2^2).
# Combined with the two perceptual quadrant constraints, this becomes 3 linear constraints on (U1, U2, Z), essentially a trivariate Gaussian orthant probability. See further details in Methods & Materials
 
 
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import owens_t
from scipy.stats import norm
 
# Number of Gauss-Legendre nodes for the 1-D integral inside trivariate_orthant. The integrand there is smooth for every parameter value, so this converges well before 64 (32 vs 64 nodes differ by ~4e-7).
tvn_nodes = 64
_tvn_z, _tvn_w = leggauss(tvn_nodes)
 
 
def ideal_limit_choice_prob(row, sens_noise1, sens_crit1, conf_noise1, conf_boost1, conf_crit1, conf_bias1, sens_noise2, sens_crit2, conf_noise2, conf_boost2, conf_crit2, conf_bias2, intrvl_bias):
 
    mu1, mu2, resp1, resp2 = row[0], row[1], row[2], row[3] # resp = perceptual judgements
 
    e1 = (sens_crit1 - mu1) / sens_noise1 # strength of sensory evidence
    e2 = (sens_crit2 - mu2) / sens_noise2
    scale1 = conf_bias1 / sens_noise1 # beta term
    scale2 = conf_bias2 / sens_noise2
 
    # a and b are the coefficients that write the confidence evidence as a linear function of standard-nomal sens_noise - see Methods & Materials section
    a1 = scale1 * (1 - conf_boost1) * sens_noise1 # coefficient on standard-normal noise, or sensory-noise sensitivity of confidence 
    b1 = scale1 * (mu1 - sens_crit1 - conf_crit1) # E[w]
    a2 = scale2 * (1 - conf_boost2) * sens_noise2
    b2 = scale2 * (mu2 - sens_crit2 - conf_crit2)
 
    s1 = 1.0 if resp1 == 1 else -1.0
    s2 = 1.0 if resp2 == 1 else -1.0
 
    # joint_prob (matches cfc_core's existing closed-form quadrant probability)
    p1 = (1 - norm.cdf(e1)) if resp1 == 1 else norm.cdf(e1)
    p2 = (1 - norm.cdf(e2)) if resp2 == 1 else norm.cdf(e2)
    joint_prob = p1 * p2
 
    # If both intervals are at full boost (a1=a2=0), all U-dependence drops out and only the confidence noise Z is left
    S = np.sqrt(conf_noise1 ** 2 + conf_noise2 ** 2)
    if abs(a1) < 1e-12 and abs(a2) < 1e-12:
        t3 = intrvl_bias + s1 * b1 - s2 * b2  # C1 - C2 - (-ib) constant, removing U-dependence
        if S > 0:
            frac = norm.cdf(t3 / S)  # only Z remains
        elif t3 > 0:
            frac = 1.0
        elif t3 < 0:
            frac = 0.0
        else:
            frac = 0.5
        return frac * joint_prob, joint_prob
 
    # Normal case, build (y1, y2, y3) as standard/linear normals
    d1 = -1.0 if resp1 == 1 else 1.0 # sign flipped so constraint becomes Y1 < bound1
    d2 = -1.0 if resp2 == 1 else 1.0
    bound1 = d1 * e1
    bound2 = d2 * e2
 
    # constraint 3: s1*a1*u1 - s2*a2*u2 > t3   where t3 = -intrvl_bias - s1*B1 + s2*B2
    t3 = -intrvl_bias - s1 * b1 + s2 * b2
    # X3 = s1*A1*U1 - s2*A2*U2 ; Y3 = -X3 (so constraint becomes y3 < -t3)
    c1 = -s1 * a1 # coefficient of U1 in Y3
    c2 = s2 * a2 # coefficient of U2 in Y3
    bound3 = -t3
 
    var3 = c1**2 + c2**2 + S**2  # + S^2 : the confidence-noise term Z. S = 0 recovers the rank-2 ideal-observer case
    cov13 = d1 * c1 # Cov(Y1, Y3) = Cov(d1*U1, c1*U1 + c2*U2 + S*Z) = d1*c1
    cov23 = d2 * c2
 
    # Standardise Y3 so all three margins are unit Normal, then evaluate the orthant probability.
    sd3 = np.sqrt(var3)
    choose1_num = trivariate_orthant(bound1, bound2, bound3 / sd3, cov13 / sd3, cov23 / sd3)
 
    return choose1_num, joint_prob
 
 
 
# Bivariate normal CDF has exact closed form via Owen's T function, while a trivariate does not. Conditioning on Y1 reduces it to a 1-dimensional integral of a bivariate, then handled by the Gauss-Legendre quadrature (integrand is smooth for every param. value, and handles the limit)
def bivariate_cdf(h, k, rho):
    # P(X < h, Y < k) for standard bivariate Normal with correlation rho (Owen's T decomposition)
    h, k, rho = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (h, k, rho)))
 
    zero_rho = np.abs(rho) < 1e-14
    out = np.where(zero_rho, norm.cdf(h) * norm.cdf(k), np.nan)  # independent case
    need = ~zero_rho
    if not np.any(need):
        return out
 
    hh, kk, rr = h[need], k[need], rho[need]
    denom = np.sqrt(np.clip(1.0 - rr * rr, 1e-300, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        a1 = (kk - rr * hh) / (hh * denom)
        a2 = (hh - rr * kk) / (kk * denom)
    # h == 0 (resp. k == 0) is the limit a -> +-inf, where Owen's T tends to +-1/4
    a1 = np.where(hh == 0.0, np.sign(kk - rr * hh) * np.inf, a1)
    a2 = np.where(kk == 0.0, np.sign(hh - rr * kk) * np.inf, a2)
    a1 = np.nan_to_num(a1, nan=0.0, posinf=1e300, neginf=-1e300)
    a2 = np.nan_to_num(a2, nan=0.0, posinf=1e300, neginf=-1e300)
 
    hk = hh * kk
    beta = np.where((hk < 0) | ((hk == 0) & (hh + kk < 0)), 0.5, 0.0)
    out[need] = np.clip(0.5 * (norm.cdf(hh) + norm.cdf(kk)) - owens_t(hh, a1) - owens_t(kk, a2) - beta, 0.0, 1.0)
    return out
 
 
def trivariate_orthant(bound1, bound2, bound3, rho13, rho23):
    # P(Y1 < bound1, Y2 < bound2, Y3 < bound3), all standard Normal, with rho12 = 0
    # (Y1 and Y2 are the two independent sensory deviates, so they are always uncorrelated).
    z, w = _tvn_z, _tvn_w
    upper = norm.cdf(bound1) # integrate Y1 over (-inf, bound1)
    p = upper * 0.5 * (z + 1.0) # probability space, finite range
    y = norm.ppf(np.clip(p, 1e-300, 1.0 - 1e-16)) # Y1 space 
    jacobian = upper * 0.5 * w # phi(y) dy = dp, density cancels
 
    # conditional on Y1 = y, Y2 and Y3 are bivariate Normal
    var13 = np.clip(1.0 - rho13 * rho13, 1e-300, None)
    cond_bound3 = (bound3 - rho13 * y) / np.sqrt(var13)
    cond_rho = np.clip(rho23 / np.sqrt(var13), -1.0 + 1e-12, 1.0 - 1e-12)
 
    inner = bivariate_cdf(np.full_like(y, bound2), cond_bound3, np.full_like(y, cond_rho))

    return float(np.clip(np.sum(jacobian * inner), 0.0, 1.0))
