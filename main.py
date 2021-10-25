import numpy as np

# ********************** Time steps management ***********************

T = 10000
N = 100
h = T/N
tab_N = [n for n in range(N)]
tab_T = [n*h for n in tab_N]


# *************************** Parameters *****************************

sigma_V = 1
V_0 = 1
kappa_V = 1
kappa_r = 1
theta_V = 1

# ************************** Initialization **************************


def initialize_v_x():
    v = [[]for n in tab_N]
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
            k_u[n].append(
                min([k_star for k_star in range(k + 1, n + 1) and v[n][k] + mu_v(v[n][k]) * h < v[n + 1][k_star]]))
            k_d[n].append(
                max([k_star for k_star in range(k + 1, n + 1) and v[n][k] + mu_v(v[n][k]) * h < v[n + 1][k_star]]))
            j_u[n].append(
                min([k_star for k_star in range(k + 1, n + 1) and v[n][k] + mu_x(x[n][k]) * h < x[n + 1][k_star]]))
            j_d[n].append(
                max([k_star for k_star in range(k + 1, n + 1) and v[n][k] + mu_x(x[n][k]) * h < x[n + 1][k_star]]))

    return k_u, k_d, j_u, j_d


# if __name__ == '__main__':
