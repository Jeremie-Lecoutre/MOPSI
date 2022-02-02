import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import random as rd

# Constants of our problem
sigma_s = 0.25    # constant stock price volatility
sigma_r = 0.08     # positive constant
kappa = 0.5         # reversion speed
S_0 = 100         # positive
r_0 = 0.06           # positive
theta = 0.1      # long term reversion target
rho = -0.25         # correlation between Z_S and Z_r
T = 1            # time to maturity
N = 50           # Number of intervals
K = 100         # Strike of the American Put Option

# The Wei and Hilliard-Schwartz-Tucker procedures
h = T/N
X_0 = np.log(S_0)/sigma_s
R_0 = 2*pow(r_0, 0.5)/sigma_r
Y_0 = (np.log(S_0)/sigma_s - 2 * rho*pow(r_0, 0.5)/sigma_r)/pow(1-pow(rho, 2), 0.5)


def mu_x(r):
    return (pow(sigma_r, 2)*pow(r, 2)/4-pow(sigma_s, 2)/2)/sigma_s


def mu_r(r):
    return (kappa*(4*theta-pow(r, 2)*pow(sigma_r, 2))-pow(sigma_r, 2))/(2*r*pow(sigma_r, 2))


def mu_y(r):
    return(mu_x(r)-rho*mu_r(r))/pow(1-pow(rho, 2), 0.5)


def mu_s(r, s):
    return r*s


# lattice construction
def init_r_y():
    r = []
    y = []
    for i in range(0, N+1):
        r_i = []
        y_i = []
        for k in range(0, i+1):
            r_i += [R_0+(2*k-i)*pow(h, 0.5)]
            y_i += [Y_0 + (2 * k - i) * pow(h, 0.5)]
        r += [r_i]
        y += [y_i]
    return r, y


R, Y, = init_r_y()


# Function to compute probabilities and movement
def k_d_i_k(i, k):
    k_d = -1
    for k_star in range(0, i+2):
        if R[i][k]+mu_r(R[i][k])*h >= R[i+1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return k + int((mu_r(R[i][k])*pow(h, 0.5)+1)/2)      # doute sur la def de int à checker

    else:
        return k_d


def k_u_i_k(i, k):
    return k_d_i_k(i, k)+1


def p_i_k(i, k):
    return max(0, min(1, (mu_r(R[i][k]) * h + R[i][k] - R[i + 1][k_d_i_k(i, k)]) / (
                R[i + 1][k_u_i_k(i, k)] - R[i + 1][k_d_i_k(i, k)])))


def j_d_i_j_k(i, j, k):
    j_d = -1
    for j_star in range(0, i+2):
        if Y[i][j]+mu_y(Y[i][k])*h >= Y[i+1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return j + int((mu_y(Y[i][k])*pow(h, 0.5)+1)/2)     # doute sur la def de int à checker
    else:
        return j_d


def j_u_i_j_k(i, j, k):
    return j_d_i_j_k(i, j, k)+1


def p_i_j_k(i, j, k):
    return max(0, min(1, (mu_y(Y[i][k]) * h + Y[i][j] - Y[i + 1][j_d_i_j_k(i, j, k)]) / (
                Y[i + 1][j_u_i_j_k(i, j, k)] - Y[i + 1][j_d_i_j_k(i, j, k)])))


# Bivariate tree
def initialize_tree():
    tree = []
    for i in range(0, N+1):
        tree_i = []
        for j in range(0, i+1):
            tree_i_j = []
            for k in range(0, i+1):
                tree_i_j += [(R[i][k], Y[i][j])]
            tree_i += [tree_i_j]
        tree += [tree_i]
    return tree


Tree = initialize_tree()


# Probability
def q_i_ju_ku(i, j, k):
    return p_i_k(i, k)*p_i_j_k(i, j, k)


def q_i_ju_kd(i, j, k):
    return (1-p_i_k(i, k))*p_i_j_k(i, j, k)


def q_i_jd_ku(i, j, k):
    return p_i_k(i, k)*(1-p_i_j_k(i, j, k))


# The robust tree algorithm

def q_i_jd_kd(i, j, k):
    return (1-p_i_k(i, k))*(1-p_i_j_k(i, j, k))


U_0 = np.log(S_0)/sigma_s


# Functions for the joint evolution of the processes r and S
def s_i_j_k(i, j, k):
    return np.exp(sigma_s*(pow(1-pow(rho, 2), 0.5)*Y[i][j] + rho * R[i][k]))


def r_i_k(i, k):
    if R[i][k] > 0:
        return pow(R[i][k]*sigma_r, 2)/4
    else:
        return 0


# lattice construction #
def initialize_lattice():
    r0 = []
    u0 = []
    for i in range(0, N+1):
        r_i = []
        u_i = []
        for k in range(0, i+1):
            r_i += [r_i_k(i, k)]
            u_i += [U_0 + (2 * k - i) * pow(h, 0.5)]
        r0 += [r_i]
        u0 += [u_i]
    return r0, u0


R0, U0 = initialize_lattice()


def k_d_new_i_k(i, k):
    k_d = -1
    for k_star in range(0, k+1):
        if R0[i][k]+mu_r(R0[i][k])*h >= R0[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return 0
    else:
        return k_d


def k_u_new_i_k(i, k):
    k_d = -1
    for k_star in range(k+1, i+2):
        if R0[i][k]+mu_r(R0[i][k])*h <= R0[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return i+1
    else:
        return k_d


def p_new_i_k(i, k):
    return max(0, min(1, (mu_r(R0[i][k]) * h + R0[i][k] - R0[i + 1][k_d_new_i_k(i, k)]) / (
            R0[i + 1][k_u_new_i_k(i, k)] - R0[i + 1][k_d_new_i_k(i, k)])))


# bivariate tree#

def initialize_tree_new():
    s_new0 = []
    tree_new0 = []
    for i in range(0, N+1):
        tree_new_i = []
        s_new_i_j = []
        for j in range(0, i+1):
            tree_new_i_j = []
            s_new_i_j += [np.exp(sigma_s * U0[i][j])]
            for k in range(0, i+1):
                tree_new_i_j += [(R0[i][k], np.exp(sigma_s * U0[i][j]))]
            tree_new_i += [tree_new_i_j]
        tree_new0 += [tree_new_i]
        s_new0 += [s_new_i_j]
    return s_new0, tree_new0


s_new, tree_new = initialize_tree_new()


def j_d_new_i_j_k(i, j, k):
    j_d = -1
    for j_star in range(0, j+1):
        if s_new[i][j]+mu_s(s_new[i][j], R0[i][k])*h >= s_new[i + 1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return 0


def j_u_new_i_j_k(i, j, k):
    j_u = -1
    for j_star in range(j+1, N+1):
        if s_new[i][j]+mu_s(s_new[i][j], R0[i][k])*h <= s_new[i + 1][j_star]:
            if j_star > j_u:
                j_u = j_star
    if j_u == -1:
        return i+1
    else:
        return j_u


# backward dynamic programming for American put option #
def initialize_v():
    v0 = []
    for j in range(0, N+1):
        v_j = []
        for k in range(0, N+1):
            v_j += [max(K-s_i_j_k(N, j, k), 0)]
        v0 += [v_j]
    return v0


v = [initialize_v()]


def update_v(v0):
    for i in range(0, N, -1):
        v_i = []
        for j in range(0, i+1):
            v_i_j = []
            for k in range(0, i+1):
                v_i_j = max(max((K - s_i_j_k(i, j, k)), 0), np.exp(-r_i_k(i, k) * h) * (
                            q_i_ju_ku(i, j, k) * v0[0][j_u_i_j_k(i, j, k)][k_u_i_k(i, k)] + q_i_ju_kd(i, j, k) *
                            v0[0][j_u_i_j_k(i, j, k)][k_d_i_k(i, k)] + q_i_jd_ku(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][
                                k_u_i_k(i, k)] + q_i_jd_kd(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][k_d_i_k(i, k)]))
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0


def p_new_i_j_k(i, j, k):
    return max(0, min(1, (mu_s(r_i_k(i, k), s_new[i][j]) * h + s_new[i][j] - s_new[i + 1][j_d_new_i_j_k(i, j, k)]) / (
                s_new[i + 1][j_u_new_i_j_k(i, j, k)] - s_new[i + 1][j_d_new_i_j_k(i, j, k)])))


def m_i_ju_ku(i, j, k):
    return (s_new[i+1][j_u_new_i_j_k(i, j, k)]-s_new[i][j])*(r_i_k(i+1, k_u_new_i_k(i, k))-r_i_k(i, k))


def m_i_jd_ku(i, j, k):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_u_new_i_k(i, k)) - r_i_k(i, k))


def m_i_ju_kd(i, j, k):
    return (s_new[i + 1][j_u_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k)) - r_i_k(i, k))


def m_i_jd_kd(i, j, k):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k)) - r_i_k(i, k))


def transition_probabilities(i, j, k):
    a = np.array([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1],
                 [m_i_ju_ku(i, j, k), m_i_ju_kd(i, j, k), m_i_jd_ku(i, j, k), m_i_jd_kd(i, j, k)])
    b = np.array([p_new_i_j_k(i, j, k), p_new_i_k(i, k), 1, rho*sigma_r*pow(r_i_k(i, k), 0.5)*sigma_s*s_new[i][j]*h])
    return np.dot(np.invert(a), b)


if __name__ == '__main__':
    print(v[0][0][0])
