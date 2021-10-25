# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# 3.2. The approximation of the component Y

def theta_r(t):     # à définir en fonction du modèle...
    return 1


def integrand_phi(t):              # intégrande de la fonction phi
    return(theta_r(t)*np.exp(kappa_r*t))

def phi(t):
    return( r_0 * np.exp(-kappa_r*t) + kappa_r * np.exp(-kappa_r*t) * integrate.quad(integrand_phi,0,t ))


def mu(v,x,t):
    return(sigma_r*x+phi(t)-eta-0.5*v-(pho_1*kappa_V*(theta_V-v)/sigma_V)+pho_2*kappa_r*x*np.sqrt(v))



# 3.3. The Monte Carlo approach
Var_delta=1 #variance des lois delta du monte carlo
Esp_delta=1 #variance des lois delta du monte carlo
n=100
delta = np.random.multivariate_normal(Esp_delta * np.eye(n), Var_delta * np.eye(n))
def Y_n_plus_1(h,X_n,X_n_plus_1,V_n,V_n_plus_1,n):
    K=np.random.poisson(Lambda*h)
    log_JK=np.random.multivariate_normal(Esp_J*np.eye(K),Var_J*np.eye(K))
    return()



