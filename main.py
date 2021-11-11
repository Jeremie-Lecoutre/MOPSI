import numpy as np
import matplotlib.pyplot as plt

# ********************** Time steps management ***********************

T = 30
N = 30
h = T/N
tab_N = range(N)
tab_T = [n*h for n in tab_N]


# *************************** Parameters *****************************

sigma_V = 1
V_0 = 1
kappa_V = 1
kappa_r = 1
theta_V = 1

# Modules:
import numpy as np
import scipy.integrate as integrate

kappa_r=1
kappa_V=1
theta_V=1
sigma_r=1
sigma_V=1
r_0=1
eta=1
pho_1=1
pho_2=1
Lambda=1
Var_J=1 #variance de log(1+J_k)
Esp_J=1 #esperance de log(1+J_k)

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
    # k_u, k_d, j_u, j_d = initialize_min_max(V,X)
    plt.subplot(1, 2, 1)
    for n in tab_N:
        for k in range(n):
            plt. scatter(n, V[n][k], s=1, color='BLACK')
    # plt.scatter(15, V[15][k_d[15, 7]], s=5, color='RED')
    # plt.scatter(15, V[15][k_d[15, 7]], s=5, color='BLUE')
    plt.subplot(1, 2, 2)
    for n in tab_N:
        for k in range(n):
            plt. scatter(n, X[n][k], s=1, color='BLACK')
    plt.show()

# 3.2. The approximation of the component Y

def theta_r(t):     # à définir en fonction du modèle...
    return 1


def integrand_phi(t):              # intégrande de la fonction phi
    return(theta_r(t)*np.exp(kappa_r*t))

def phi(t):
    return( r_0 * np.exp(-kappa_r*t) + kappa_r * np.exp(-kappa_r*t) * integrate.quad(integrand_phi,0,t ))


def mu(v,x,t):
    return(sigma_r*x+phi(t)-eta-0.5*v-(pho_1*kappa_V*(theta_V-v)/sigma_V)+pho_2*kappa_r*x*np.sqrt(v))



# 3.3. The Monte Carlo approach for the approximation of the component Y
Var_delta=1 #variance des lois delta du monte carlo
Esp_delta=1 #variance des lois delta du monte carlo
n=100
vec_1=np.arange(n)
np.ones_like(vec_1)
delta = np.random.multivariate_normal(Esp_delta*vec_1, Var_delta * np.eye(n))
def pho_3(pho_1,pho_2):
    return (np.sqrt((1-pho_1**2-pho_2**2)))
def Y_n_plus_1(h,Y_n,X_n,X_n_plus_1,V_n,V_n_plus_1,_n):
    K=np.random.poisson(Lambda*h)
    somme_log_JK =0
    if(K>0):
        vector_k=np.arange(K)
        np.ones_like(vector_k)
        log_JK=np.random.multivariate_normal(Esp_J*vector_k,Var_J*np.eye(K))
        somme_log_JK=np.sum(log_JK)
    Y_n_plus_un= Y_n + mu(V_n,X_n,_n*h)+ pho_3(pho_1,pho_2)*np.sqrt(h*V_n)*delta[_n+1]+pho_1*(V_n_plus_1-V_n)/sigma_V + pho_2*np.sqrt(V_n)*(X_n_plus_1-X_n)+somme_log_JK
    return(Y_n_plus_un)



