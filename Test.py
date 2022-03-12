import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import scipy.stats as si
import random as rd
import csv

sigma_s = 0.25
tab_sigma_r = [0.08]
# tab_sigma_r = [0.08, 0.5, 1, 3]
kappa = 0.5
S_0 = 100
r_0 = 0.06
theta = 0.1
rho = -0.25
tab_T = [1]
# tab_T = [1, 2]
tab_N = [50]
# tab_N = [50, 100, 150, 200, 300]
K = 100


def constantes(sigma_r, n, t):
    h = t / n
    x_0 = np.log(S_0) / sigma_s
    r0 = 2 * pow(r_0, 0.5) / sigma_r
    y_0 = (np.log(S_0) / sigma_s - 2 * rho * pow(r_0, 0.5) / sigma_r) / pow(1 - pow(rho, 2), 0.5)
    u_0 = np.log(S_0) / sigma_s
    return h, x_0, r0, y_0, u_0


def mu_x(r, sigma_r):
    return (pow(sigma_r, 2) * pow(r, 2) / 4 - pow(sigma_s, 2) / 2) / sigma_s


def mu_r(r, sigma_r):
    return (kappa * (4 * theta - pow(r, 2) * pow(sigma_r, 2)) - pow(sigma_r, 2)) / (2 * r * pow(sigma_r, 2))


def mu_y(r, sigma_r):
    return (mu_x(r, sigma_r) - rho * mu_r(r, sigma_r)) / pow(1 - pow(rho, 2), 0.5)


def init_r_y(n, r0, y0, h):
    r = []
    y = []
    for i in range(0, n + 1):
        r_i = []
        y_i = []
        for k in range(0, i + 1):
            r_i += [r0 + (2 * k - i) * pow(h, 0.5)]
            y_i += [y0 + (2 * k - i) * pow(h, 0.5)]
        r += [r_i]
        y += [y_i]
    return r, y


def init_r_y_optimised(n, r0, y_0, h):
    r = np.zeros((n + 1, n + 1))
    y = np.zeros((n + 1, n + 1))
    for i in range(0, n + 1):
        for k in range(0, i + 1):
            r[i][k] = r0 + (2 * k - i) * pow(h, 0.5)
            y[i][k] = y_0 + (2 * k - i) * pow(h, 0.5)
    return r, y


# Function to compute probabilities and movement
def k_d_i_k(i, k, h, sigma_r, r):
    k_d = -1
    for k_star in range(0, i + 1):
        if r[i][k] + mu_r(r[i][k], sigma_r) * h >= r[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return i

    else:
        return k_d


def k_d_i_k2(i, k, h, sigma_r, r):
    return int(k + np.floor((mu_r(r[i][k], sigma_r) * pow(h, 1 / 2) + 1) / 2))


def k_u_i_k(i, k, h, sigma_r, r):
    return k_d_i_k(i, k, h, sigma_r, r) + 1


def p_i_k(i, k, h, sigma_r, r):
    return max(0, min(1, (mu_r(r[i][k], sigma_r) * h + r[i][k] - r[i + 1][k_d_i_k(i, k, h, sigma_r, r)]) / (
            r[i + 1][k_u_i_k(i, k, h, sigma_r, r)] - r[i + 1][k_d_i_k(i, k, h, sigma_r, r)])))


def j_d_i_j_k(i, j, k, h, sigma_r, r):
    j_d = -1
    for j_star in range(0, i + 1):
        if Y[i][j] + mu_y(r[i][k], sigma_r) * h >= Y[i + 1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return i
    else:
        return j_d


def j_d_i_j_k2(i, j, k, h, sigma_r, r):
    return int(j + np.floor((mu_y(r[i][k], sigma_r) * pow(h, 1 / 2) + 1) / 2))


def j_u_i_j_k(i, j, k, h, sigma_r, r):
    return j_d_i_j_k(i, j, k, h, sigma_r, r) + 1


def p_i_j_k(i, j, k, h, sigma_r, r, y):
    return max(0, min(1, (mu_y(r[i][k], sigma_r) * h + y[i][j] - y[i + 1][j_d_i_j_k(i, j, k, h, sigma_r, r)]) / (
            y[i + 1][j_u_i_j_k(i, j, k, h, sigma_r, r)] - y[i + 1][j_d_i_j_k(i, j, k, h, sigma_r, r)])))


# Ploting the different lattice and movement upon them
def plot_lattice_movement_r(i, k, n, h, sigma_r, r):
    for l in range(0, n + 1):
        for m in range(0, l+1):
            plt.scatter(l, r[l][m], s=1, color='BLACK')
    plt.scatter(i + 1, r[i + 1][k_u_i_k(i, k, h, sigma_r, r)], s=20, marker='o', color='BLUE')
    plt.scatter(i, r[i][k], s=20, marker='^', color='GREEN')
    plt.scatter(i + 1, r[i + 1][k_d_i_k(i, k, h, sigma_r, r)], s=20, marker='o', color='RED')
    plt.title("Mouvement sur la lattice de R")
    plt.xlabel("temps")
    plt.show()
    return 0


def plot_lattice_movement_y(i, j, k, n, h, sigma_r, r, y):
    for l in range(0, n + 1):
        for m in range(0, l+1):
            plt.scatter(l, y[l][m], s=1, color='BLACK')
    plt.scatter(i + 1, y[i + 1][j_u_i_j_k(i, j, k, h, sigma_r, r)], s=20, marker='o', color='BLUE')
    plt.scatter(i, y[i][j], s=20, marker='^', color='GREEN')
    plt.scatter(i + 1, y[i + 1][j_d_i_j_k(i, j, k, h, sigma_r, r)], s=20, marker='o', color='RED')
    plt.title("Mouvement sur la lattice de Y")
    plt.xlabel("temps")
    plt.show()
    return 0


# Bivariate tree
def initialize_tree(n, r, y):
    tree = []
    for i in range(0, n + 1):
        tree_i = []
        for j in range(0, i + 1):
            tree_i_j = []
            for k in range(0, i + 1):
                tree_i_j += [(r[i][k], y[i][j])]
            tree_i += [tree_i_j]
        tree += [tree_i]
    return tree


def q_i_ju_ku(i, j, k, h, sigma_r, r, y):
    return p_i_k(i, k, h, sigma_r, r) * p_i_j_k(i, j, k, h, sigma_r, r, y)


def q_i_ju_kd(i, j, k, h, sigma_r, r, y):
    return (1 - p_i_k(i, k, h, sigma_r, r)) * p_i_j_k(i, j, k, h, sigma_r, r, y)


def q_i_jd_ku(i, j, k, h, sigma_r, r, y):
    return p_i_k(i, k, h, sigma_r, r) * (1 - p_i_j_k(i, j, k, h, sigma_r, r, y))


def q_i_jd_kd(i, j, k, h, sigma_r, r, y):
    return (1 - p_i_k(i, k, h, sigma_r, r)) * (1 - p_i_j_k(i, j, k, h, sigma_r, r, y))


# Functions for the joint evolution of the processes r and S
def s_i_j_k(i, j, k, r, y):
    return np.exp(sigma_s * (pow(1 - pow(rho, 2), 0.5) * y[i][j] + rho * r[i][k]))


def r_i_k(i, k, sigma_r, r):
    if r[i][k] > 0:
        return pow(r[i][k] * sigma_r, 2) / 4
    else:
        return 0


def initialize_v(n, r, y):
    v0 = []
    for j in range(0, n + 1):
        v_j = []
        for k in range(0, n + 1):
            v_j += [max(K - s_i_j_k(n, j, k, r, y), 0)]
        v0 += [v_j]
    return v0


def update_v(v0, n, h, sigma_r, r, y):
    for i in range(n - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                v_i_j += [max(max((K - s_i_j_k(i, j, k, r, y)), 0), np.exp(-r_i_k(i, k, sigma_r, r) * h) * (
                        q_i_ju_ku(i, j, k, h, sigma_r, r, y) * v0[0][j_u_i_j_k(i, j, k, h, sigma_r, r)][k_u_i_k(i, k, h, sigma_r, r)] + q_i_ju_kd(i, j, k, h, sigma_r, r, y) *
                        v0[0][j_u_i_j_k(i, j, k, h, sigma_r, r)][k_d_i_k(i, k, h, sigma_r, r)] + q_i_jd_ku(i, j, k, h, sigma_r, r, y) * v0[0][j_d_i_j_k(i, j, k, h, sigma_r, r)][
                            k_u_i_k(i, k, h, sigma_r, r)] + q_i_jd_kd(i, j, k, h, sigma_r, r, y) * v0[0][j_d_i_j_k(i, j, k, h, sigma_r, r)][k_d_i_k(i, k, h, sigma_r, r)]))]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0


def initialize_v_euro(n, r, y):
    v0 = []
    for j in range(0, n + 1):
        v_j = []
        for k in range(0, n + 1):
            v_j += [max(K - s_i_j_k(n, j, k, r, y), 0)]
        v0 += [v_j]
    return v0


def update_v_euro(v0, n, h, sigma_r, r, y):
    for i in range(n - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                v_i_j += [np.exp(-r_i_k(i, k, sigma_r, r) * h) * (
                        q_i_ju_ku(i, j, k, h, sigma_r, r, y) * v0[0][j_u_i_j_k(i, j, k, h, sigma_r, r)][k_u_i_k(i, k, h, sigma_r, r)] + q_i_ju_kd(i, j, k, h, sigma_r, r, y) *
                        v0[0][j_u_i_j_k(i, j, k, h, sigma_r, r)][k_d_i_k(i, k, h, sigma_r, r)] + q_i_jd_ku(i, j, k, h, sigma_r, r, y) * v0[0][j_d_i_j_k(i, j, k, h, sigma_r, r)][
                            k_u_i_k(i, k, h, sigma_r, r)] + q_i_jd_kd(i, j, k, h, sigma_r, r, y) * v0[0][j_d_i_j_k(i, j, k, h, sigma_r, r)][k_d_i_k(i, k, h, sigma_r, r)])]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0


def jump(i, j, k, h, sigma_r, r, y):
    p = rd.random()
    q_sum = q_i_jd_kd(i, j, k, h, sigma_r, r, y)
    if p < q_sum:
        return s_i_j_k(i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r)
    if q_sum < p < q_sum + q_i_jd_ku(i, j, k, h, sigma_r, r, y):
        return s_i_j_k(i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r)
    q_sum += q_i_jd_ku(i, j, k, h, sigma_r, r, y)
    if q_sum < p < q_sum + q_i_ju_kd(i, j, k, h, sigma_r, r, y):
        return s_i_j_k(i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r)
    q_sum += q_i_ju_kd(i, j, k, h, sigma_r, r, y)
    if q_sum < p < q_sum + q_i_ju_ku(i, j, k, h, sigma_r, r, y):
        return s_i_j_k(i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r)


def simulation(n, h, sigma_r, r, y):
    data = [S_0]
    i = 0
    j = 0
    k = 0
    while i < n:
        s, i, j, k = jump(i, j, k, h, sigma_r, r, y)
        data += [s]
    return data


def plot_simulation(n, h, sigma_r, r, y):
    data = simulation(n, h, sigma_r, r, y)
    for i in range(0, len(data)):
        plt.scatter(i, data[i], s=1, color='RED')


def new_mu_r(r):
    return kappa * (theta - r)


def mu_s(r, s):
    return r * s


def initialize_lattice(n, h, sigma_r, u_0, r):
    r0 = []
    u0 = []
    for i in range(0, n + 1):
        r_i = []
        u_i = []
        for k in range(0, i + 1):
            r_i += [r_i_k(i, k, sigma_r, r)]
            u_i += [u_0 + (2 * k - i) * pow(h, 0.5)]
        r0 += [r_i]
        u0 += [u_i]
    return r0, u0


def initialize_tree_new(n, r_new, u_new):
    s_new0 = []
    tree_new0 = []
    for i in range(0, n + 1):
        tree_new_i = []
        s_new_i_j = []
        for j in range(0, i + 1):
            tree_new_i_j = []
            s_new_i_j += [np.exp(sigma_s * u_new[i][j])]
            for k in range(0, i + 1):
                tree_new_i_j += [(r_new[i][k], np.exp(sigma_s * u_new[i][j]))]
            tree_new_i += [tree_new_i_j]
        tree_new0 += [tree_new_i]
        s_new0 += [s_new_i_j]
    return s_new0, tree_new0


def k_d_new_i_k(i, k, h, r_new):
    k_d = -1
    for k_star in range(0, k + 1):
        if r_new[i][k] + new_mu_r(r_new[i][k]) * h >= r_new[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return 0
    else:
        return k_d


def k_u_new_i_k(i, k, h, r_new):
    k_u = i + 1
    for k_star in range(k + 1, i + 2):
        if r_new[i][k] + new_mu_r(r_new[i][k]) * h <= r_new[i + 1][k_star]:
            if k_star < k_u:
                k_u = k_star
    if k_u == -1:
        return i + 1
    else:
        return k_u


def j_d_new_i_j_k(i, j, k, h, r_new, s_new):
    j_d = -1
    for j_star in range(0, j + 1):
        if s_new[i][j] + mu_s(r_new[i][k], s_new[i][j]) * h >= s_new[i + 1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return 0
    else:
        return j_d


def j_u_new_i_j_k(i, j, k, h, r_new, s_new):
    j_u = i + 1
    for j_star in range(j + 1, i + 2):
        if s_new[i][j] + mu_s(r_new[i][k], s_new[i][j]) * h <= s_new[i + 1][j_star]:
            if j_star < j_u:
                j_u = j_star
    if j_u == -1:
        return i + 1
    else:
        return j_u


def p_new_i_k(i, k, h, r_new):
    return max(0, min(1, (new_mu_r(r_new[i][k]) * h + r_new[i][k] - r_new[i + 1][k_d_new_i_k(i, k, h, r_new)]) / (
            r_new[i + 1][k_u_new_i_k(i, k, h, r_new)] - r_new[i + 1][k_d_new_i_k(i, k, h, r_new)])))


def p_new_i_j_k(i, j, k, h, sigma_r, r_new, s_new):
    return max(0, min(1, (mu_s(r_i_k(i, k, sigma_r, r_new), s_new[i][j]) * h + s_new[i][j] - s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)]) / (
            s_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)] - s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)])))


def new_plot_lattice_movement_r0(i, k, n, h, r_new):
    for l in range(0, n + 1):
        for m in range(0, l+1):
            plt.scatter(l, r_new[l][m], s=1, color='BLACK')
    plt.scatter(i + 1, r_new[i + 1][k_u_new_i_k(i, k, h, r_new)], s=20, marker='o', color='BLUE')
    plt.scatter(i, r_new[i][k], s=20, marker='^', color='GREEN')
    plt.scatter(i + 1, r_new[i + 1][k_d_new_i_k(i, k, h, r_new)], s=20, marker='o', color='RED')
    plt.title("Mouvement sur la lattice de R0")
    plt.xlabel("temps")
    plt.show()
    return 0


def plot_lattice_movement_u0(i, j, k, n, h, r_new, u_new, s_new):
    for l in range(0, n + 1):
        for m in range(0, l+1):
            plt.scatter(l, u_new[l][m], s=1, color='BLACK')
    plt.scatter(i + 1, u_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)], s=20, marker='o', color='BLUE')
    plt.scatter(i, u_new[i][j], s=20, marker='^', color='GREEN')
    plt.scatter(i + 1, u_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)], s=20, marker='o', color='RED')
    plt.title("Mouvement sur la lattice de U0")
    plt.xlabel("temps")
    plt.show()
    return 0


def m_i_ju_ku(i, j, k, sigma_r, h, r_new, s_new):
    return (s_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)] - s_new[i][j]) * (r_i_k(i + 1, k_u_new_i_k(i, k, h, r_new), sigma_r, r_new) - r_i_k(i, k, sigma_r, r_new))


def m_i_jd_ku(i, j, k, sigma_r, h, r_new, s_new):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)] - s_new[i][j]) * (r_i_k(i + 1, k_u_new_i_k(i, k, h, r_new), sigma_r, r_new) - r_i_k(i, k, sigma_r, r_new))


def m_i_ju_kd(i, j, k, sigma_r, h, r_new, s_new):
    return (s_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k, h, r_new), sigma_r, r_new) - r_i_k(i, k, sigma_r, r_new))


def m_i_jd_kd(i, j, k, sigma_r, h, r_new, s_new):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k, h, r_new), sigma_r, r_new) - r_i_k(i, k, sigma_r, r_new))


def transition_probabilities(i, j, k, sigma_r, h, r_new, s_new):
    a = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1],
                  [m_i_ju_ku(i, j, k, sigma_r, h, r_new, s_new), m_i_ju_kd(i, j, k, sigma_r, h, r_new, s_new), m_i_jd_ku(i, j, k, sigma_r, h, r_new, s_new), m_i_jd_kd(i, j, k, sigma_r, h, r_new, s_new)]])
    b = np.array(
        [p_new_i_j_k(i, j, k, h, sigma_r, r_new, s_new), p_new_i_k(i, k, h, r_new), 1, rho * sigma_r * pow(r_i_k(i, k, sigma_r, r_new), 0.5) * sigma_s * s_new[i][j] * h])
    return alg.solve(a, b)


def initialize_v_new(n, s_new):
    v0 = []
    for j in range(0, n + 1):
        v_j = []
        for k in range(0, n + 1):
            v_j += [max(K - s_new[n][j], 0)]
        v0 += [v_j]
    return v0


def update_v_new(v0, n, sigma_r, h, r_new, s_new):
    for i in range(n - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                probability = transition_probabilities(i, j, k, sigma_r, h, r_new, s_new)
                q_i_ju_ku0 = probability[0]
                q_i_ju_kd0 = probability[1]
                q_i_jd_ku0 = probability[2]
                q_i_jd_kd0 = probability[3]
                v_i_j += [max(max((K - s_new[i][j]), 0), np.exp(-r_i_k(i, k, sigma_r, r_new) * h) * (
                        q_i_ju_ku0 * v0[0][j_u_new_i_j_k(i, j, k, h, r_new, s_new)][k_u_new_i_k(i, k, h, r_new)] + q_i_ju_kd0 *
                        v0[0][j_u_new_i_j_k(i, j, k, h, r_new, s_new)][k_d_new_i_k(i, k, h, r_new)] + q_i_jd_ku0 *
                        v0[0][j_d_new_i_j_k(i, j, k, h, r_new, s_new)][k_u_new_i_k(i, k, h, r_new)] + q_i_jd_kd0 *
                        v0[0][j_d_new_i_j_k(i, j, k, h, r_new, s_new)][k_d_new_i_k(i, k, h, r_new)]))]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0


def initialize_v_new_euro(n, s_new):
    v0 = []
    for j in range(0, n + 1):
        v_j = []
        for k in range(0, n + 1):
            v_j += [max(K - s_new[n][j], 0)]
        v0 += [v_j]
    return v0


def update_v_new_euro(v0, n, sigma_r, h, r_new, s_new):
    for i in range(n - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                probability = transition_probabilities(i, j, k, sigma_r, h, r_new, s_new)
                q_i_ju_ku0 = probability[0]
                q_i_ju_kd0 = probability[1]
                q_i_jd_ku0 = probability[2]
                q_i_jd_kd0 = probability[3]
                v_i_j += [np.exp(-r_i_k(i, k, sigma_r, r_new) * h) * (
                        q_i_ju_ku0 * v0[0][j_u_new_i_j_k(i, j, k, h, r_new, s_new)][k_u_new_i_k(i, k, h, r_new)] + q_i_ju_kd0 *
                        v0[0][j_u_new_i_j_k(i, j, k, h, r_new, s_new)][k_d_new_i_k(i, k, h, r_new)] + q_i_jd_ku0 *
                        v0[0][j_d_new_i_j_k(i, j, k, h, r_new, s_new)][k_u_new_i_k(i, k, h, r_new)] + q_i_jd_kd0 *
                        v0[0][j_d_new_i_j_k(i, j, k, h, r_new, s_new)][k_d_new_i_k(i, k, h, r_new)])]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0


def plot_tree(n, v):
    for i in range(0, n + 1):
        for j in range(0, i+1):
            for k in range(0, i+1):
                plt.scatter(i, v[i][j][k], s=1, color='BLACK')
    plt.show()
    return 0


def plot_ku_kd(n, h, sigma_r, r_new):
    for i in range(0, n):
        for k in range(0, i + 1):
            plt.subplot(1, 2, 1)
            plt.scatter(i, k_u_i_k(i, k, h, sigma_r, r_new), s=1, color='BLUE', label='k_u')
            plt.subplot(1, 2, 2)
            plt.scatter(i, k_d_i_k(i, k, h, sigma_r, r_new), s=1, color='RED', label='k_d')
    plt.show()
    return 0


def new_jump(i, j, k, sigma_r, h, r_new, s_new):
    p = rd.random()
    probability = transition_probabilities(i, j, k, sigma_r, h, r_new, s_new)
    q_i_ju_ku0 = probability[0]
    q_i_ju_kd0 = probability[1]
    q_i_jd_ku0 = probability[2]
    q_i_jd_kd0 = probability[3]
    q_sum = q_i_ju_ku0
    if p < q_sum:
        return s_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)], i + 1, j_u_new_i_j_k(i, j, k, h, r_new, s_new), k_u_new_i_k(i, k, h, r_new)
    if q_sum < p < q_sum + q_i_ju_kd0:
        return s_new[i + 1][j_u_new_i_j_k(i, j, k, h, r_new, s_new)], i + 1, j_u_new_i_j_k(i, j, k, h, r_new, s_new), k_d_new_i_k(i, k, h, r_new)
    q_sum += q_i_ju_kd0
    if q_sum < p < q_sum + q_i_jd_ku0:
        return s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)], i + 1, j_d_new_i_j_k(i, j, k, h, r_new, s_new), k_u_new_i_k(i, k, h, r_new)
    q_sum += q_i_jd_ku0
    if q_sum < p < q_sum + q_i_jd_kd0:
        return s_new[i + 1][j_d_new_i_j_k(i, j, k, h, r_new, s_new)], i + 1, j_d_new_i_j_k(i, j, k, h, r_new, s_new), k_d_new_i_k(i, k, h, r_new)


def new_simulation(n, sigma_r, h, r_new, s_new):
    data = [S_0]
    i = 0
    j = 0
    k = 0
    while i < n:
        s, i, j, k = new_jump(i, j, k, sigma_r, h, r_new, s_new)
        data += [s]
    return data


def new_plot_simulation(n, sigma_r, h, r_new, s_new):
    data = new_simulation(n, sigma_r, h, r_new, s_new)
    for i in range(0, len(data)):
        plt.scatter(i, data[i], s=1, color='GREEN')


def monte_carlo_approach(simulation_number, t, n, sigma_r):
    r_i = r_0*np.ones(simulation_number)
    s_i = S_0*np.ones(simulation_number)
    theta_tab = theta * np.ones(simulation_number)
    for i in range(1, n + 1):
        gaussian_vector1 = np.random.multivariate_normal(np.zeros(simulation_number), np.eye(simulation_number))
        gaussian_vector2 = np.random.multivariate_normal(np.zeros(simulation_number), np.eye(simulation_number))
        r_i_plus_1 = r_i + kappa * (theta_tab - r_i) * t / n + sigma_r * pow(r_i * t / n, 0.5) * gaussian_vector1
        s_i_plus_1 = s_i * np.exp((r_i-0.5*(sigma_s**2)*np.ones(simulation_number)) * t / n + sigma_s * pow(t / n, 0.5) * (rho * gaussian_vector1 + pow(1 - rho ** 2, 0.5) * gaussian_vector2))
        r_i = r_i_plus_1
        s_i = s_i_plus_1
    for i in range(simulation_number):
        s_i[i] = max(0, K-s_i[i])/simulation_number
        r_i[i] = r_i[i]/simulation_number
    return r_i.sum(), s_i.sum()


def jump_mc(i, j, k, h, sigma_r, r, y):
    p = rd.random()
    q_sum = q_i_jd_kd(i, j, k, h, sigma_r, r, y)
    if p < q_sum:
        return r_i_k(i + 1, k_d_i_k(i, k, h, sigma_r, r), sigma_r, r), s_i_j_k(i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r)
    if q_sum < p < q_sum + q_i_jd_ku(i, j, k, h, sigma_r, r, y):
        return r_i_k(i + 1, k_u_i_k(i, k, h, sigma_r, r), sigma_r, r), s_i_j_k(i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_d_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r)
    q_sum += q_i_jd_ku(i, j, k, h, sigma_r, r, y)
    if q_sum < p < q_sum + q_i_ju_kd(i, j, k, h, sigma_r, r, y):
        return r_i_k(i + 1, k_d_i_k(i, k, h, sigma_r, r), sigma_r, r), s_i_j_k(i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_d_i_k(i, k, h, sigma_r, r)
    q_sum += q_i_ju_kd(i, j, k, h, sigma_r, r, y)
    if q_sum < p < q_sum + q_i_ju_ku(i, j, k, h, sigma_r, r, y):
        return r_i_k(i + 1, k_u_i_k(i, k, h, sigma_r, r), sigma_r, r), s_i_j_k(i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r), r, y), i + 1, j_u_i_j_k(i, j, k, h, sigma_r, r), k_u_i_k(i, k, h, sigma_r, r)


def simulation_mc(n, h, sigma_r, r, y):
    s = S_0
    r_mc = r_0
    i = 0
    j = 0
    k = 0
    while i < n:
        r_mc, s, i, j, k = jump_mc(i, j, k, h, sigma_r, r, y)
    return r_mc, s


def mc_tree(nb_simul, n, h, sigma_r, r, y):
    tab_r = []
    tab_s = []
    for i in range(nb_simul):
        r_mc, s = simulation_mc(n, h, sigma_r, r, y)
        tab_r.append(r_mc)
        tab_s.append(max(0, K-s))
    return np.array(tab_r).sum()/nb_simul, np.array(tab_s).sum()/nb_simul


def black_scholes_put_option(t):
    d1 = (np.log(S_0 / K) + (r_0 + 0.5 * sigma_s ** 2) * t) / (sigma_s * np.sqrt(t))
    d2 = (np.log(S_0 / K) + (r_0 - 0.5 * sigma_s ** 2) * t) / (sigma_s * np.sqrt(t))

    put = (K * np.exp(-r_0 * t) * si.norm.cdf(-d2, 0.0, 1.0) - S_0 * si.norm.cdf(-d1, 0.0, 1.0))
    return put


if __name__ == '__main__':
    fichier = open('resultats.csv', 'w')
    ecrivainCSV = csv.writer(fichier, delimiter=";")
    ecrivainCSV.writerow(
        ["Paramètres", "Wei and Hilliard Amer", "Wei and Hilliard Euro", "Robust Tree Americaine", "Robust Tree Euro",
         "Simple Monte-Carlo Euro", "Monte-Carlo Tree Euro"])

    for T in tab_T:
        for Sigma_r in tab_sigma_r:
            for N in tab_N:

                H, X_0, R_0, Y_0, U_0 = constantes(Sigma_r, N, T)

                R, Y = init_r_y(N, R_0, Y_0, H)
                Tree = initialize_tree(N, R, Y)

                R_new, U_new = initialize_lattice(N, H, Sigma_r, U_0, R)
                S_new, tree_new = initialize_tree_new(N, R_new, U_new)

                WH_Amer = update_v([initialize_v(N, R, Y)], N, H, Sigma_r, R, Y)[0][0][0]
                WH_Euro = update_v_euro([initialize_v_euro(N, R, Y)], N, H, Sigma_r, R, Y)[0][0][0]

                Tree_Amer = update_v_new([initialize_v_new(N, S_new)], N, Sigma_r, H, R_new, S_new)[0][0][0]
                Tree_Euro = update_v_new_euro([initialize_v_new_euro(N, S_new)], N, Sigma_r, H, R_new, S_new)[0][0][0]

                r_MC, s_MC = monte_carlo_approach(1000, T, N, Sigma_r)
                r_MC_tree, s_MC_tree = mc_tree(1000, T, N, Sigma_r, R, Y)

                #cv2.imwrite(plot_simulation(N, Y, R, sigma_r, h), plot_simulation(N, Y, R, sigma_r, h))
                #cv2.imwrite(plot_ku_kd(R, sigma_r, h, T, N), plot_ku_kd(R, sigma_r, h, T, N))
                #cv2.imwrite(new_plot_simulation(N, T, S_0, R0, s_new, h, R, sigma_r),
                #            new_plot_simulation(N, T, S_0, R0, s_new, h, R, sigma_r))

                ecrivainCSV.writerow(["T = " + str(T) + "; sigma_R = " + str(Sigma_r) + "; N = " + str(N), str(WH_Amer), str(WH_Euro), str(Tree_Amer), str(Tree_Euro), str(s_MC), str(s_MC_tree)])

    fichier.close()