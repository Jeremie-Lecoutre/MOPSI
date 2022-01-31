import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random as rd

# ********************** Time steps management ***********************

T = 30
N = 30
h = T / N
tab_N = range(N+1)
tab_T = [n * h for n in tab_N]

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
        for k in range(n+1):
            v_k_n = 0
            if np.sqrt(V_0) + (sigma_V / 2) * (2 * k - n) * np.sqrt(h) > 0:
                v_k_n = (np.sqrt(V_0) + (sigma_V / 2) * (2 * k - n) * np.sqrt(h)) ** 2
            v[n].append(v_k_n)
            x[n].append((2 * k - n) * np.sqrt(h))
    return v, x


def initialize_v_x_2():
    v = []
    x = []
    for n in tab_N:
        v_n = []
        x_n = []
        for k in range(n+1):
            if np.sqrt(V_0) + (sigma_V / 2) * (2 * k - n) * np.sqrt(h) > 0:
                v_n.append((np.sqrt(V_0) + (sigma_V / 2) * (2 * k - n) * np.sqrt(h)) ** 2)
            else:
                v_n.append(0)
            x_n.append((2 * k - n) * np.sqrt(h))
        v.append(v_n)
        x.append(x_n)
    del(v[0])
    del(x[0])
    return v, x


def mu_v(v):
    return kappa_V * (theta_V - v)


def mu_x(x):
    return -kappa_r * x


def initialize_min_max(v, x):
    k_u = [[] for n in tab_N]
    k_d = [[] for n in tab_N]
    j_u = [[] for n in tab_N]
    j_d = [[] for n in tab_N]
    for n in tab_N[0:len(tab_N)-1]:
        for k in range(n+1):
            k_u_inter = []
            k_d_inter = []
            j_u_inter = []
            j_d_inter = []
            for counter in range(k + 1, n+2):
                if v[n][k] + mu_v(v[n][k]) * h <= v[n+1][counter]:
                    k_u_inter.append(counter)
            for counter in range(k+1):
                if v[n][k] + mu_v(v[n][k]) * h >= v[n+1][counter]:
                    k_d_inter.append(counter)
            for counter in range(k + 1, n+2):
                if x[n][k] + mu_x(x[n][k]) * h <= x[n+1][counter]:
                    j_u_inter.append(counter)
            for counter in range(k+1):
                if x[n][k] + mu_x(x[n][k]) * h >= x[n+1][counter]:
                    j_d_inter.append(counter)
            if not k_u_inter:
                k_u[n].append(n+1)
            else:
                k_u[n].append(min(k_u_inter))
            if not k_d_inter:
                k_d[n].append(0)
            else:
                k_d[n].append(max(k_d_inter))
            if not j_u_inter:
                j_u[n].append(n+1)
            else:
                j_u[n].append(min(j_u_inter))
            if not j_d_inter:
                j_d[n].append(0)
            else:
                j_d[n].append(max(j_d_inter))
    return k_u, k_d, j_u, j_d


def initialize_probability(v, x, k_u, k_d, j_u, j_d):
    p_u_v = [[] for n in tab_N]
    p_d_v = [[] for n in tab_N]
    p_u_x = [[] for n in tab_N]
    p_d_x = [[] for n in tab_N]
    for n in tab_N[0:len(tab_N)-2]:
        for k in range(n+1):
            p_u_v[n].append(max(0, min(1, (mu_v(v[n][k]) + v[n][k] - v[n + 1][k_d[n][k]]) / (
                    v[n + 1][k_u[n][k]] - v[n + 1][k_d[n][k]]))))
            p_u_x[n].append(max(0, min(1, (mu_x(x[n][k]) + x[n][k] - x[n + 1][j_d[n][k]]) / (
                    x[n + 1][j_u[n][k]] - x[n + 1][j_d[n][k]]))))
            p_d_v[n].append(1 - p_u_v[n][k])
            p_d_x[n].append(1 - p_u_x[n][k])
    return p_u_v, p_d_v, p_u_x, p_d_x


def puu(n, k, j, p_u_v, p_u_x):
    return p_u_v[n][k] * p_u_x[n][j]


def pud(n, k, j, p_u_v, p_d_x):
    return p_u_v[n][k] * p_d_x[n][j]


def pdu(n, k, j, p_d_v, p_u_x):
    return p_d_v[n][k] * p_u_x[n][j]


def pdd(n, k, j, p_d_v, p_d_x):
    return p_d_v[n][k] * p_d_x[n][j]


def jump(n, k, j, k_u, k_d, j_u, j_d, p_u_v, p_u_x, p_d_v, p_d_x):
    p = rd.random()
    p_inter = puu(n, k, j, p_u_v, p_u_x)
    print( "1 =", pud(n, k, j, p_u_v, p_d_x)+pdu(n, k, j, p_d_v, p_u_x)+puu(n, k, j, p_u_v, p_u_x)+ pdd(n, k, j, p_d_v, p_d_x))
    if p < p_inter:
        return n+1, k_u[n][k], j_u[n][j]
    if p_inter < p < p_inter + pud(n, k, j, p_u_v, p_d_x):
        return n+1, k_u[n][k], j_d[n][j]
    p_inter += pud(n, k, j, p_u_v, p_d_x)
    if p_inter < p < p_inter + pdu(n, k, j, p_d_v, p_u_x):
        return n+1, k_d[n][k], j_u[n][j]
    p_inter += pdu(n, k, j, p_d_v, p_u_x)
    if p_inter < p < p_inter + pdd(n, k, j, p_d_v, p_d_x):
        return n+1, k_d[n][k], j_d[n][j]

    #else:
    #    return n+1, k, j


def simulation():
    v, x = initialize_v_x()
    k_u, k_d, j_u, j_d = initialize_min_max(V, X)
    result_v = [v[0][0]]
    result_x = [x[0][0]]
    n, k, j = 0, 0, 0
    p_u_v, p_d_v, p_u_x, p_d_x = initialize_probability(v, x, k_u, k_d, j_u, j_d)
    for counter in range(N-1):
        n, k, j = jump(n, k, j, k_u, k_d, j_u, j_d, p_u_v, p_u_x, p_d_v, p_d_x)
        result_v.append(v[n][k])
        result_x.append(x[n][j])
    return result_v, result_x


if __name__ == '__main__':
    V, X = initialize_v_x()
    k_u_test, k_d_test, j_u_test, j_d_test = initialize_min_max(V, X)
    plt.subplot(1, 2, 1)
    for nn in tab_N:
        for kk in range(nn):
            plt.scatter(nn, V[nn][kk], s=1, color='BLACK')
    plt.subplot(1, 2, 2)
    for nn in tab_N:
        for kk in range(nn):
            plt.scatter(nn, X[nn][kk], s=1, color='BLACK')
    plt.show()
    result_V, result_X = simulation()
    print(result_V)
    print(result_X)


# ************************** New parameters ****************************

kappa_r = 1
kappa_V = 1
theta_V = 1
sigma_r = 1
sigma_V = 1
r_0 = 1
eta = 1
pho_1 = 1
pho_2 = 1
Lambda = 1
Var_J = 1  # variance of log(1+J_k)
Esp_J = 1  # mean of log(1+J_k)


# 3.2. The approximation of the component Y

def theta_r(t):  # to be defined depending on the model
    return 1


def integrand_phi(t):  # integrand of the function phi
    return theta_r(t) * np.exp(kappa_r * t)


def phi(t):
    return r_0 * np.exp(-kappa_r * t) + kappa_r * np.exp(-kappa_r * t) * integrate.quad(integrand_phi, 0, t)

# 4. The Hybrid Tree/Finite Difference Approach
def mu(v, x, t):
    return sigma_r * x + phi(t) - eta - 0.5 * v - (
            pho_1 * kappa_V * (theta_V - v) / sigma_V) + pho_2 * kappa_r * x * np.sqrt(v)


# 3.3. The Monte Carlo approach for the approximation of the component Y

Var_delta = 1  # Monte Carlo delta laws variance
Esp_delta = 1  # Monte Carlo delta laws mean
n = 100
vec_1 = np.arange(n)
np.ones_like(vec_1)
delta = np.random.multivariate_normal(Esp_delta * vec_1, Var_delta * np.eye(n))


def pho_3(pho_1_test, pho_2_test):
    return np.sqrt((1 - pho_1_test ** 2 - pho_2_test ** 2))


def y_n_plus_1(y_n, x_n, x_n_plus_1, v_n, v_n_plus_1, _n):
    k = np.random.poisson(Lambda * h)
    somme_log_jk = 0
    y_n_plus_un = 0
    if k > 0:
        vector_k = np.arange(k)
        np.ones_like(vector_k)
        log_jk = np.random.multivariate_normal(Esp_J * vector_k, Var_J * np.eye(k))
        somme_log_jk = np.sum(log_jk)
        y_n_plus_un = y_n + mu(v_n, x_n, _n * h) + pho_3(pho_1, pho_2) * np.sqrt(h * v_n) * delta[_n + 1] + pho_1 * (
                v_n_plus_1 - v_n) / sigma_V + pho_2 * np.sqrt(v_n) * (x_n_plus_1 - x_n) + somme_log_jk
    return y_n_plus_un

# 4. The Hybrid Tree/Finite Difference Approach
# 4.1. The local 1-dimensional partial integro-differential equation
# 4.1.1. Finite-difference and numerical quadrature

M=500 #qui doit être entier et supérieur à R
R=100
Y0=0.15
delta_y=0.1
grille_finie_Y_M=Y0*np.ones(2*M)+delta_y*np.linspace(-M,M,2*M)

def A(n,v,x):
    alpha= h*mu(v,x,n*h)/(2*delta_y)
    beta= h*(pho_3(pho_1,pho_2)**2)*v/(2*(delta_y**2))
    A=np.eye(2*M+1)
    for i in range(1,2*M):
        A[i][i-1]=alpha-beta
        A[i-1][i]=-alpha-beta
    return A
def densite_loi_normale(x,esperance,ecart_type):
    return(np.exp(-0.5*(((x-esperance)/ecart_type)**2))/(ecart_type*np.sqrt(2*np.pi)))

gamma=0                             #valeur implémentée juste après
for i in range(-R,R,1):
    gamma+=Lambda*densite_loi_normale(i*delta_y,Esp_J,Var_J)

B=np.eye(2*M+1)+h*delta_y*(Lambda*densite_loi_normale(0,Esp_J,Var_J)-gamma)*np.eye(2*M+1)
for i in range(1, 2 * M):
    B[i][i - 1] += h*delta_y*Lambda*densite_loi_normale(-delta_y,Esp_J,Var_J)
    B[i - 1][i] += h*delta_y*Lambda*densite_loi_normale(delta_y,Esp_J,Var_J)

# ici b(nh,y_i) vaut 0 car les indices sont inférieurs à R ou M
