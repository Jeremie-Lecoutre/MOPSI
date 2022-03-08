import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import scipy.stats as si
import random as rd

sigma_s = 0.25
sigma_r = 0.08
kappa = 0.5
S_0 = 100
r_0 = 0.06
theta = 0.1
rho = -0.25
T = 1
N = 50
K = 100
h = T / N
X_0 = np.log(S_0) / sigma_s
R_0 = 2 * pow(r_0, 0.5) / sigma_r
Y_0 = (np.log(S_0) / sigma_s - 2 * rho * pow(r_0, 0.5) / sigma_r) / pow(1 - pow(rho, 2), 0.5)
U_0 = np.log(S_0) / sigma_s

def mu_x(r):
    return (pow(sigma_r, 2) * pow(r, 2) / 4 - pow(sigma_s, 2) / 2) / sigma_s


def mu_r(r):
    return (kappa * (4 * theta - pow(r, 2) * pow(sigma_r, 2)) - pow(sigma_r, 2)) / (2 * r * pow(sigma_r, 2))


def mu_y(r):
    return (mu_x(r) - rho * mu_r(r)) / pow(1 - pow(rho, 2), 0.5)


def init_r_y():
    r = []
    y = []
    for i in range(0, N + 1):
        r_i = []
        y_i = []
        for k in range(0, i + 1):
            r_i += [R_0 + (2 * k - i) * pow(h, 0.5)]
            y_i += [Y_0 + (2 * k - i) * pow(h, 0.5)]
        r += [r_i]
        y += [y_i]
    return r, y


R, Y, = init_r_y()

# Function to compute probabilities and movement
def k_d_i_k(i, k):
    k_d = -1
    for k_star in range(0, i + 1):
        if R[i][k] + mu_r(R[i][k]) * h >= R[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return i

    else:
        return k_d


def k_d_i_k2(i, k):
    return int(k + np.floor((mu_r(R[i][k]) * pow(h, 1 / 2) + 1) / 2))


def k_u_i_k(i, k):
    return k_d_i_k(i, k) + 1


def p_i_k(i, k):
    return max(0, min(1, (mu_r(R[i][k]) * h + R[i][k] - R[i + 1][k_d_i_k(i, k)]) / (
            R[i + 1][k_u_i_k(i, k)] - R[i + 1][k_d_i_k(i, k)])))


def j_d_i_j_k(i, j, k):
    j_d = -1
    for j_star in range(0, i + 1):
        if Y[i][j] + mu_y(R[i][k]) * h >= Y[i + 1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return i
    else:
        return j_d


def j_d_i_j_k2(i, j, k):
    return int(j + np.floor((mu_y(R[i][k]) * pow(h, 1 / 2) + 1) / 2))


def j_u_i_j_k(i, j, k):
    return j_d_i_j_k(i, j, k) + 1


def p_i_j_k(i, j, k):
    return max(0, min(1, (mu_y(R[i][k]) * h + Y[i][j] - Y[i + 1][j_d_i_j_k(i, j, k)]) / (
            Y[i + 1][j_u_i_j_k(i, j, k)] - Y[i + 1][j_d_i_j_k(i, j, k)])))


# Ploting the different lattice and movement upon them
def plot_lattice_movement_r(i,k):
    for l in range(0, N+1):
        for m in range(0,l+1):
            plt.scatter(l, R[l][m], s=1, color='BLACK')
    plt.scatter(i+1, R[i+1][k_u_i_k(i,k)], s=20,marker='o', color ='BLUE')
    plt.scatter(i, R[i][k], s=20, marker ='^', color='GREEN')
    plt.scatter(i+1, R[i+1][k_d_i_k(i, k)], s=20,marker='o', color='RED')
    plt.title("Mouvement sur la lattice de R")
    plt.xlabel("temps")
    plt.show()
    return 0

def plot_lattice_movement_y(i,j,k):
    for l in range(0, N+1):
        for m in range(0,l+1):
            plt.scatter(l, Y[l][m], s=1, color='BLACK')
    plt.scatter(i+1, Y[i+1][j_u_i_j_k(i,j,k)], s=20,marker='o', color ='BLUE')
    plt.scatter(i, Y[i][j], s=20, marker ='^',color='GREEN')
    plt.scatter(i+1, Y[i+1][j_d_i_j_k(i,j,k)], s=20,marker='o' , color='RED')
    plt.title("Mouvement sur la lattice de Y")
    plt.xlabel("temps")
    plt.show()
    return 0

# Bivariate tree
def initialize_tree():
    tree = []
    for i in range(0, N + 1):
        tree_i = []
        for j in range(0, i + 1):
            tree_i_j = []
            for k in range(0, i + 1):
                tree_i_j += [(R[i][k], Y[i][j])]
            tree_i += [tree_i_j]
        tree += [tree_i]
    return tree


Tree = initialize_tree()

def q_i_ju_ku(i, j, k):
    return p_i_k(i, k) * p_i_j_k(i, j, k)


def q_i_ju_kd(i, j, k):
    return (1 - p_i_k(i, k)) * p_i_j_k(i, j, k)


def q_i_jd_ku(i, j, k):
    return p_i_k(i, k) * (1 - p_i_j_k(i, j, k))


def q_i_jd_kd(i, j, k):
    return (1 - p_i_k(i, k)) * (1 - p_i_j_k(i, j, k))

# Functions for the joint evolution of the processes r and S
def s_i_j_k(i, j, k):
    return np.exp(sigma_s * (pow(1 - pow(rho, 2), 0.5) * Y[i][j] + rho * R[i][k]))


def r_i_k(i, k):
    if R[i][k] > 0:
        return pow(R[i][k] * sigma_r, 2) / 4
    else:
        return 0

def initialize_v():
    v0 = []
    for j in range(0, N + 1):
        v_j = []
        for k in range(0, N + 1):
            v_j += [max(K - s_i_j_k(N, j, k), 0)]
        v0 += [v_j]
    return v0


def update_v(v0):
    for i in range(N - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                v_i_j += [max(max((K - s_i_j_k(i, j, k)), 0), np.exp(-r_i_k(i, k) * h) * (
                        q_i_ju_ku(i, j, k) * v0[0][j_u_i_j_k(i, j, k)][k_u_i_k(i, k)] + q_i_ju_kd(i, j, k) *
                        v0[0][j_u_i_j_k(i, j, k)][k_d_i_k(i, k)] + q_i_jd_ku(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][
                            k_u_i_k(i, k)] + q_i_jd_kd(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][k_d_i_k(i, k)]))]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0

def v_optimised_2():
    v = np.zeros((N+1,N+1,N+1))
    mat_k = K*np.ones((N+1, N+1))
    mat_s = [[s_i_j_k(N,j,k) for j in range(0, N+1)]for k in range(0, N+1)]
    v[N] = (mat_k - mat_s)
    v[N][v[N] < 0] = 0
    v[N] = v[N].transpose()
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            for k in range(0, i + 1):
              j_u=j_u_i_j_k(i, j, k)
              j_d=j_d_i_j_k(i, j, k)
              k_u=k_u_i_k(i, k)
              k_d=k_d_i_k(i, k)
              v[i][j][k]= max(max((K - s_i_j_k(i, j, k)), 0), np.exp(-r_i_k(i, k) * h) * (
                        q_i_ju_ku(i, j, k) * v[i+1][j_u][k_u] + q_i_ju_kd(i, j, k) *
                        v[i+1][j_u][k_d] + q_i_jd_ku(i, j, k) * v[i+1][j_d][
                            k_u] + q_i_jd_kd(i, j, k) * v[i+1][j_d][k_d]))
    return v
v_optimised_2()[0][0][0]

# benchmark the time it takes
# %timeit v_optimised()
v=v_optimised_2()
print(v[0][0][0])

def initialize_v_euro():
    v0 = []
    for j in range(0, N + 1):
        v_j = []
        for k in range(0, N + 1):
            v_j += [max(K - s_i_j_k(N, j, k), 0)]
        v0 += [v_j]
    return v0


def update_v_euro(v0):
    for i in range(N - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                v_i_j += [ np.exp(-r_i_k(i, k) * h) * (
                        q_i_ju_ku(i, j, k) * v0[0][j_u_i_j_k(i, j, k)][k_u_i_k(i, k)] + q_i_ju_kd(i, j, k) *
                        v0[0][j_u_i_j_k(i, j, k)][k_d_i_k(i, k)] + q_i_jd_ku(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][
                            k_u_i_k(i, k)] + q_i_jd_kd(i, j, k) * v0[0][j_d_i_j_k(i, j, k)][k_d_i_k(i, k)])]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0

v_euro = update_v_euro([initialize_v_euro()])
print(v_euro[0][0][0])

def jump(i, j, k):
    p = rd.random()
    q_sum = q_i_jd_kd(i, j, k)
    if p < q_sum:
        return s_i_j_k(i + 1, j_d_i_j_k(i, j, k), k_d_i_k(i, k)), i + 1, j_d_i_j_k(i, j, k), k_d_i_k(i, k)
    if q_sum < p < q_sum + q_i_jd_ku(i, j, k):
        return s_i_j_k(i + 1, j_d_i_j_k(i, j, k), k_u_i_k(i, k)), i + 1, j_d_i_j_k(i, j, k), k_u_i_k(i, k)
    q_sum += q_i_jd_ku(i, j, k)
    if q_sum < p < q_sum + q_i_ju_kd(i, j, k):
        return s_i_j_k(i + 1, j_u_i_j_k(i, j, k), k_d_i_k(i, k)), i + 1, j_u_i_j_k(i, j, k), k_d_i_k(i, k)
    q_sum += q_i_ju_kd(i, j, k)
    if q_sum < p < q_sum + q_i_ju_ku(i, j, k):
        return s_i_j_k(i + 1, j_u_i_j_k(i, j, k), k_u_i_k(i, k)), i + 1, j_u_i_j_k(i, j, k), k_u_i_k(i, k)


def simulation():
    data = [S_0]
    i = 0
    j = 0
    k = 0
    while i < N:
        s, i, j, k = jump(i, j, k)
        data += [s]
    return data


def plot_simulation():
    data = simulation()
    for i in range(0, len(data)):
        plt.scatter(i, data[i], s=1, color='RED')

def new_mu_r(r):
    return kappa * (theta - r)


def mu_s(r, s):
    return r * s


def initialize_lattice():
    r0 = []
    u0 = []
    for i in range(0, N + 1):
        r_i = []
        u_i = []
        for k in range(0, i + 1):
            r_i += [r_i_k(i, k)]
            u_i += [U_0 + (2 * k - i) * pow(h, 0.5)]
        r0 += [r_i]
        u0 += [u_i]
    return r0, u0

R0, U0 = initialize_lattice()

def k_d_new_i_k(i, k):
    k_d = -1
    for k_star in range(0, k + 1):
        if R0[i][k] + new_mu_r(R0[i][k]) * h >= R0[i + 1][k_star]:
            if k_star > k_d:
                k_d = k_star
    if k_d == -1:
        return 0
    else:
        return k_d


def k_u_new_i_k(i, k):
    k_u = i + 1
    for k_star in range(k + 1, i + 2):
        if R0[i][k] + new_mu_r(R0[i][k]) * h <= R0[i + 1][k_star]:
            if k_star < k_u:
                k_u = k_star
    if k_u == -1:
        return i + 1
    else:
        return k_u


def j_d_new_i_j_k(i, j, k):
    j_d = -1
    for j_star in range(0, j + 1):
        if s_new[i][j] + mu_s(s_new[i][j], R0[i][k]) * h >= s_new[i + 1][j_star]:
            if j_star > j_d:
                j_d = j_star
    if j_d == -1:
        return 0
    else:
        return j_d


def j_u_new_i_j_k(i, j, k):
    j_u = i + 1
    for j_star in range(j + 1, i + 2):
        if s_new[i][j] + mu_s(s_new[i][j], R0[i][k]) * h <= s_new[i + 1][j_star]:
            if j_star < j_u:
                j_u = j_star
    if j_u == -1:
        return i + 1
    else:
        return j_u


def p_new_i_k(i, k):
    return max(0, min(1, (new_mu_r(R0[i][k]) * h + R0[i][k] - R0[i + 1][k_d_new_i_k(i, k)]) / (
            R0[i + 1][k_u_new_i_k(i, k)] - R0[i + 1][k_d_new_i_k(i, k)])))


def p_new_i_j_k(i, j, k):
    return max(0, min(1, (mu_s(r_i_k(i, k), s_new[i][j]) * h + s_new[i][j] - s_new[i + 1][j_d_new_i_j_k(i, j, k)]) / (
            s_new[i + 1][j_u_new_i_j_k(i, j, k)] - s_new[i + 1][j_d_new_i_j_k(i, j, k)])))


def new_plot_lattice_movement_r0(i,k):
    for l in range(0, N+1):
        for m in range(0,l+1):
            plt.scatter(l, R0[l][m], s=1, color='BLACK')
    plt.scatter(i+1, R0[i+1][k_u_new_i_k(i,k)], s=20,marker='o', color ='BLUE')
    plt.scatter(i, R0[i][k], s=20, marker ='^', color='GREEN')
    plt.scatter(i+1, R0[i+1][k_d_new_i_k(i, k)], s=20,marker='o', color='RED')
    plt.title("Mouvement sur la lattice de R0")
    plt.xlabel("temps")
    plt.show()
    return 0

def plot_lattice_movement_u0(i,j,k):
    for l in range(0, N+1):
        for m in range(0,l+1):
            plt.scatter(l, U0[l][m], s=1, color='BLACK')
    plt.scatter(i+1, U0[i+1][j_u_new_i_j_k(i,j,k)], s=20,marker='o', color ='BLUE')
    plt.scatter(i, U0[i][j], s=20, marker ='^',color='GREEN')
    plt.scatter(i+1, U0[i+1][j_d_new_i_j_k(i,j,k)], s=20,marker='o' , color='RED')
    plt.title("Mouvement sur la lattice de U0")
    plt.xlabel("temps")
    plt.show()
    return 0

def m_i_ju_ku(i, j, k):
    return (s_new[i + 1][j_u_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_u_new_i_k(i, k)) - r_i_k(i, k))


def m_i_jd_ku(i, j, k):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_u_new_i_k(i, k)) - r_i_k(i, k))


def m_i_ju_kd(i, j, k):
    return (s_new[i + 1][j_u_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k)) - r_i_k(i, k))


def m_i_jd_kd(i, j, k):
    return (s_new[i + 1][j_d_new_i_j_k(i, j, k)] - s_new[i][j]) * (r_i_k(i + 1, k_d_new_i_k(i, k)) - r_i_k(i, k))


def transition_probabilities(i, j, k):
    a = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1],
                  [m_i_ju_ku(i, j, k), m_i_ju_kd(i, j, k), m_i_jd_ku(i, j, k), m_i_jd_kd(i, j, k)]])
    b = np.array(
        [p_new_i_j_k(i, j, k), p_new_i_k(i, k), 1, rho * sigma_r * pow(r_i_k(i, k), 0.5) * sigma_s * s_new[i][j] * h])
    return alg.solve(a, b)

def initialize_tree_new():
    s_new0 = []
    tree_new0 = []
    for i in range(0, N + 1):
        tree_new_i = []
        s_new_i_j = []
        for j in range(0, i + 1):
            tree_new_i_j = []
            s_new_i_j += [np.exp(sigma_s * U0[i][j])]
            for k in range(0, i + 1):
                tree_new_i_j += [(R0[i][k], np.exp(sigma_s * U0[i][j]))]
            tree_new_i += [tree_new_i_j]
        tree_new0 += [tree_new_i]
        s_new0 += [s_new_i_j]
    return s_new0, tree_new0


s_new, tree_new = initialize_tree_new()

def initialize_v_new():
    v0 = []
    for j in range(0, N + 1):
        v_j = []
        for k in range(0, N + 1):
            v_j += [max(K - s_new[N][j], 0)]
        v0 += [v_j]
    return v0


def update_v_new(v0):
    for i in range(N - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                probability = transition_probabilities(i, j, k)
                q_i_ju_ku0 = probability[0]
                q_i_ju_kd0 = probability[1]
                q_i_jd_ku0 = probability[2]
                q_i_jd_kd0 = probability[3]
                v_i_j += [max(max((K - s_new[i][j]), 0), np.exp(-r_i_k(i, k) * h) * (
                        q_i_ju_ku0 * v0[0][j_u_new_i_j_k(i, j, k)][k_u_new_i_k(i, k)] + q_i_ju_kd0 *
                        v0[0][j_u_new_i_j_k(i, j, k)][k_d_new_i_k(i, k)] + q_i_jd_ku0 *
                        v0[0][j_d_new_i_j_k(i, j, k)][k_u_new_i_k(i, k)] + q_i_jd_kd0 *
                        v0[0][j_d_new_i_j_k(i, j, k)][k_d_new_i_k(i, k)]))]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0

#Attention, exécution de l'ordre de 20 secondes
v_new = [initialize_v_new()]
v_new = update_v_new(v_new)
print("Prix de l'option de vente américaine : " + str(v_new[0][0][0]))

def initialize_v_new_euro():
    v0 = []
    for j in range(0, N + 1):
        v_j = []
        for k in range(0, N + 1):
            v_j += [max(K - s_new[N][j], 0)]
        v0 += [v_j]
    return v0


def update_v_new_euro(v0):
    for i in range(N - 1, -1, -1):
        v_i = []
        for j in range(0, i + 1):
            v_i_j = []
            for k in range(0, i + 1):
                probability = transition_probabilities(i, j, k)
                q_i_ju_ku0 = probability[0]
                q_i_ju_kd0 = probability[1]
                q_i_jd_ku0 = probability[2]
                q_i_jd_kd0 = probability[3]
                v_i_j += [np.exp(-r_i_k(i, k) * h) * (
                        q_i_ju_ku0 * v0[0][j_u_new_i_j_k(i, j, k)][k_u_new_i_k(i, k)] + q_i_ju_kd0 *
                        v0[0][j_u_new_i_j_k(i, j, k)][k_d_new_i_k(i, k)] + q_i_jd_ku0 *
                        v0[0][j_d_new_i_j_k(i, j, k)][k_u_new_i_k(i, k)] + q_i_jd_kd0 *
                        v0[0][j_d_new_i_j_k(i, j, k)][k_d_new_i_k(i, k)])]
            v_i += [v_i_j]
        v0 = [v_i] + v0
    return v0

#Attention, exécution de l'ordre de 20 secondes
v_new_euro = [initialize_v_new_euro()]
v_new_euro = update_v_new_euro(v_new_euro)
print("Prix de l'option européenne par robust tree : " + str(v_new_euro[0][0][0]))

def plot_tree():
    for i in range(0, N+1):
        for j in range(0, i+1):
            for k in range(0, i+1):
                plt.scatter(i, v[i][j][k], s=1, color='BLACK')
    plt.show()
    return 0



def plot_ku_kd():
    for i in range(0, N):
        for k in range(0, i + 1):
            plt.subplot(1, 2, 1)
            plt.scatter(i, k_u_i_k(i, k), s=1, color='BLUE', label='k_u')
            plt.subplot(1, 2, 2)
            plt.scatter(i, k_d_i_k(i, k), s=1, color='RED', label='k_d')
    plt.show()
    return 0

def new_jump(i, j, k):
    p = rd.random()
    probability = transition_probabilities(i, j, k)
    q_i_ju_ku0 = probability[0]
    q_i_ju_kd0 = probability[1]
    q_i_jd_ku0 = probability[2]
    q_i_jd_kd0 = probability[3]
    q_sum = q_i_ju_ku0
    if p < q_sum:
        return s_new[i + 1][j_u_new_i_j_k(i, j, k)], i + 1, j_u_new_i_j_k(i, j, k), k_u_new_i_k(i, k)
    if q_sum < p < q_sum + q_i_ju_kd0:
        return s_new[i + 1][j_u_new_i_j_k(i, j, k)], i + 1, j_u_new_i_j_k(i, j, k), k_d_new_i_k(i, k)
    q_sum += q_i_ju_kd0
    if q_sum < p < q_sum + q_i_jd_ku0:
        return s_new[i + 1][j_d_new_i_j_k(i, j, k)], i + 1, j_d_new_i_j_k(i, j, k), k_u_new_i_k(i, k)
    q_sum += q_i_jd_ku0
    if q_sum < p < q_sum + q_i_jd_kd0:
        return s_new[i + 1][j_d_new_i_j_k(i, j, k)], i + 1, j_d_new_i_j_k(i, j, k), k_d_new_i_k(i, k)


def new_simulation():
    data = [S_0]
    i = 0
    j = 0
    k = 0
    while i < N:
        s, i, j, k = new_jump(i, j, k)
        data += [s]
    return data


def new_plot_simulation():
    data = new_simulation()
    for i in range(0, len(data)):
        plt.scatter(i, data[i], s=1, color='GREEN')

def Monte_carlo_approach(simulation_number):
  r_i = r_0*np.ones(simulation_number)
  S_i = S_0*np.ones(simulation_number)
  theta_tab = theta *np.ones(simulation_number)
  for i in range(1,N+1):
    gaussian_vector1=np.random.multivariate_normal(np.zeros(simulation_number), np.eye(simulation_number))
    gaussian_vector2=np.random.multivariate_normal(np.zeros(simulation_number), np.eye(simulation_number))
    r_i_plus_1 = r_i + kappa*(theta_tab - r_i)*T/N + sigma_r*pow(r_i*T/N,0.5)*gaussian_vector1
    S_i_plus_1 = S_i* np.exp((r_i-0.5*(sigma_s**2)*np.ones(simulation_number))*T/N + sigma_s *pow(T/N,0.5)*(rho*gaussian_vector1 + pow(1-rho**2,0.5)*gaussian_vector2))
    r_i = r_i_plus_1
    S_i = S_i_plus_1
  for i in range(simulation_number):
    S_i[i] = max(0, K-S_i[i])/simulation_number
    r_i[i] = r_i[i]/simulation_number
  return r_i.sum(), S_i.sum()

def jump_MC(i, j, k):
    p = rd.random()
    q_sum = q_i_jd_kd(i, j, k)
    if p < q_sum:
        return r_i_k(i + 1, k_d_i_k(i, k)), s_i_j_k(i + 1, j_d_i_j_k(i, j, k), k_d_i_k(i, k)), i + 1, j_d_i_j_k(i, j, k), k_d_i_k(i, k)
    if q_sum < p < q_sum + q_i_jd_ku(i, j, k):
        return r_i_k(i + 1, k_u_i_k(i, k)), s_i_j_k(i + 1, j_d_i_j_k(i, j, k), k_u_i_k(i, k)), i + 1, j_d_i_j_k(i, j, k), k_u_i_k(i, k)
    q_sum += q_i_jd_ku(i, j, k)
    if q_sum < p < q_sum + q_i_ju_kd(i, j, k):
        return r_i_k(i + 1, k_d_i_k(i, k)), s_i_j_k(i + 1, j_u_i_j_k(i, j, k), k_d_i_k(i, k)), i + 1, j_u_i_j_k(i, j, k), k_d_i_k(i, k)
    q_sum += q_i_ju_kd(i, j, k)
    if q_sum < p < q_sum + q_i_ju_ku(i, j, k):
        return r_i_k(i + 1, k_u_i_k(i, k)), s_i_j_k(i + 1, j_u_i_j_k(i, j, k), k_u_i_k(i, k)), i + 1, j_u_i_j_k(i, j, k), k_u_i_k(i, k)


def simulation():
    S = S_0
    r = r_0
    i = 0
    j = 0
    k = 0
    while i < N:
        r, s, i, j, k = jump_MC(i, j, k)
    return r, s

def MC_tree(nb_simul):
  tab_r = []
  tab_s = []
  for i in range(nb_simul):
    r, s = simulation()
    tab_r.append(r)
    tab_s.append(max(0,K-s))
  return np.array(tab_r).sum()/nb_simul, np.array(tab_s).sum()/nb_simul

r_MC, s_MC = Monte_carlo_approach(1000)
print("Prix de l'option européenne par Monte-Carlo simple : " + str(s_MC))

r_MC_tree, s_MC_tree = MC_tree(1000)
print("Prix de l'option européenne par Monte-Carlo et la méthode d'arbres : "+ str(s_MC_tree))

#Attention : exécution de l'ordre de 30 secondes
if __name__ == '__main__':
  """
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            for k in range(0, i + 1):
                plt.scatter(i, v[i][j][k], s=1, color='BLACK')
    plt.show()"""
  plot_simulation()
  plt.show()
  new_plot_simulation()
  plt.show()
  plot_lattice_movement_r(25, 9)
  plot_lattice_movement_y(25, 9, 15)
  new_plot_lattice_movement_r0(35, 17)
  plot_lattice_movement_u0(35, 17, 15)
