import numpy as np
import matplotlib.pyplot as plt

# ********************** Time steps management ***********************

T = 30
N = 30
h = T/N
tab_N = range(N)
tab_T = [n*h for n in tab_N]


# *************************** Parameters *****************************

sigma_V = 0.4
V_0 = 0.04
kappa_V = 2
kappa_r = 2
theta_V = 0.04

# ************************** Initialization **************************


def initialize_v_x():
    v = [[] for n in tab_N]
    x = [[] for n in tab_N]
    for n in tab_N:
        for k in range(n):
            v_k_n = 0
            if np.sqrt(V_0)+sigma_V/2*(2*k-n)*np.sqrt(h) > 0:
                v_k_n = (np.sqrt(V_0)+sigma_V/2*(2*k-n)*np.sqrt(h))**2
            v[n].append(v_k_n)
            x[n].append((2*k-n)*np.sqrt(h))
    return v, x


def mu_v(v):
    return kappa_V*(theta_V-v)


def mu_x(x):
    return -kappa_r*x


def initialize_min_max(v, x):
    k_u = [[] for n in tab_N]
    k_d = [[] for n in tab_N]
    j_u = [[] for n in tab_N]
    j_d = [[] for n in tab_N]
    for n in tab_N:
        for k in range(n):
            k_u_inter = []
            k_d_inter = []
            j_u_inter = []
            j_d_inter = []
            for counter in range(k+1, n): # Remettre les bons indices/effets de bords
                if v[n][k] + mu_v(v[n][k]) * h <= v[n][counter]:
                    k_u_inter.append(counter)
            for counter in range(k):
                if v[n][k] + mu_v(v[n][k]) * h >= v[n][counter]:
                    k_d_inter.append(counter)
            for counter in range(k+1, n):
                if x[n][k] + mu_x(x[n][k]) * h <= x[n][counter]:
                    j_u_inter.append(counter)
            for counter in range(k):
                if x[n][k] + mu_x(x[n][k]) * h >= x[n][counter]:
                    j_d_inter.append(counter)

            if not k_u_inter:
                k_u[n].append(-1)
            else:
                k_u[n].append(min(k_u_inter))
            if not k_d_inter:
                k_d[n].append(-1)
            else:
                k_d[n].append(min(k_d_inter))
            if not j_u_inter:
                j_u[n].append(-1)
            else:
                j_u[n].append(min(j_u_inter))
            if not j_d_inter:
                j_d[n].append(-1)
            else:
                j_d[n].append(min(j_d_inter))
    print(k_d)
    return k_u, k_d, j_u, j_d


def initialize_probability(v, x, k_u, k_d, j_u, j_d):
    p_u_v = [[] for n in tab_N]
    p_d_v = [[] for n in tab_N]
    p_u_x = [[] for n in tab_N]
    p_d_x = [[] for n in tab_N]
    for n in tab_N:
        for k in range(n):
            p_u_v[n].append(max(0, min(1, (mu_v(v[n][k]) + v[n][k] - v[n + 1][k_d[n][k]]) / (
                        v[n + 1][k_u[n][k]] - v[n + 1][k_d[n][k]]))))
            p_u_x[n].append(max(0, min(1, (mu_x(x[n][k]) + x[n][k] - x[n + 1][j_d[n][k]]) / (
                        x[n + 1][j_u[n][k]] - x[n + 1][j_d[n][k]]))))
    for n in tab_N:
        for k in range(n):
            p_d_v[n].append(1-p_u_v[n][k])
            p_d_x[n].append(1-p_u_x[n][k])
    return p_u_v, p_d_v, p_u_x, p_d_x


def puu(n, k, j, p_u_v, p_u_x):
    return p_u_v[n][k] * p_u_x[n][j]


def pud(n, k, j, p_u_v, p_d_x):
    return p_u_v[n][k] * p_d_x[n][j]


def pdu(n, k, j, p_d_v, p_u_x):
    return p_d_v[n][k] * p_u_x[n][j]


def pdd(n, k, j, p_d_v, p_d_x):
    return p_d_v[n][k] * p_d_x[n][j]


if __name__ == '__main__':
    V, X = initialize_v_x()
    k_u_test, k_d_test, j_u_test, j_d_test = initialize_min_max(V, X)
    plt.subplot(1, 2, 1)
    for nn in tab_N:
        for kk in range(nn):
            plt. scatter(nn, V[nn][kk], s=1, color='BLACK')
    plt.scatter(15, V[15][k_d_test[15][7]], s=5, color='RED')
    plt.scatter(15, V[15][k_d_test[15][7]], s=5, color='BLUE')
    plt.subplot(1, 2, 2)
    for nn in tab_N:
        for kk in range(nn):
            plt. scatter(nn, X[nn][kk], s=1, color='BLACK')
    plt.show()
