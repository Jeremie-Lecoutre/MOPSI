import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random as rd

##### Constant of our problem #####
sigma_s= 0.25    # constant stock price volatility
sigma_r= 0.3     # positive constant
kappa= 3         # reversion speed
S_0= 145         # positive
r_0= 2           # positive
theta = 0.2      # long term reversion target
rho= 0.4         # correlation between Z_S and Z_r
T = 1            # time to maturity
N = 30           # Number of intervals
K = 3325         # Strike of the American Put Option

##### The Wei and Hilliard-Schwartz-Tucker procedures #####
h=T/N
X_0 = np.log(S_0)/sigma_s
R_0 = 2*pow(r_0,0.5)/sigma_r
Y_0 = (np.log(S_0)/sigma_s -2*rho*pow(r_0,0.5)/sigma_r)/pow(1-pow(rho,2),0.5)

def mu_X(R):
    return((pow(sigma_r,2)*pow(R,2)/4-pow(sigma_s,2)/2)/sigma_s)

def mu_R(R):
    return((kappa*(4*theta-pow(R,2)*pow(sigma_r,2))-pow(sigma_r,2))/(2*R*pow(sigma_r,2)))

def mu_Y(R):
    return((mu_X(R)-rho*mu_R(R))/pow(1-pow(rho,2),0.5))

# lattice construction #
R=[]
Y=[]
R_i=[]
Y_i=[]
for i in range(0,N+1):
    R_i = []
    Y_i = []
    for k in range(0,i+1):
        R_i+=[R_0+(2*k-i)*pow(h,0.5)]
        Y_i += [Y_0 + (2 * k - i) * pow(h, 0.5)]
    R+=[R_i]
    Y+=[Y_i]

#Function to compute probabilities and movement
def k_d_i_k(i,k):
    k_d=-1
    for k_star in range(0,i+2):
        if(R[i][k]+mu_R(R[i][k])*h >= R[i+1][k_star]):
            if(k_star>k_d):
                k_d= k_star
    if(k_d == -1):
        return(k +int((mu_R(R[i][k])*pow(h,0.5)+1)/2))      #doute sur la def de int à checker
    else:
        return(k_d)

def k_u_i_k(i,k):
    return(k_d_i_k(i,k)+1)

def p_i_k(i,k):
    return(max(0,min(1,(mu_R(R[i][k])*h + R[i][k]- R[i+1][k_d_i_k(i,k)])/(R[i+1][k_u_i_k(i,k)]-R[i+1][k_d_i_k(i,k)]))))


def j_d_i_j_k(i,j,k):
    j_d=-1
    for j_star in range(0,i+2):
        if(Y[i][j]+mu_Y(Y[i][k])*h >= R[i+1][j_star]):
            if(j_star>j_d):
                j_d= j_star
    if(j_d == -1):
        return(j +int((mu_Y(Y[i][k])*pow(h,0.5)+1)/2))      #doute sur la def de int à checker
    else:
        return(j_d)

def j_u_i_j_k(i,j,k):
    return(j_d_i_j_k(i,j,k)+1)

def p_i_j_k(i,j,k):
    return(max(0,min(1,(mu_Y(Y[i][k])*h + Y[i][j]- Y[i+1][j_d_i_j_k(i,j,k)])/(Y[i+1][j_u_i_j_k(i,j,k)]-Y[i+1][j_d_i_j_k(i,j,k)]))))

# bivariate tree#

Tree=[]
Tree_i=[]
Tree_i_j=[]

for i in range(0,N+1):
    Tree_i=[]
    for j in range(0,i+1):
        Tree_i_j=[]
        for k in range(0,i+1):
            Tree_i_j+=[(R[i][k],Y[i][j])]
        Tree_i+=[Tree_i_j]
    Tree+=[Tree_i]

# Probability #

def q_i_ju_ku(i,j,k):
    return (p_i_k(i,k)*p_i_j_k(i,j,k))

def q_i_ju_kd(i,j,k):
    return ((1-p_i_k(i,k))*p_i_j_k(i,j,k))

def q_i_jd_ku(i,j,k):
    return(p_i_k(i,k)*(1-p_i_j_k(i,j,k)))

def q_i_jd_kd(i,j,k):
    return((1-p_i_k(i,k))*(1-p_i_j_k(i,j,k)))

# Functions for the joint evolution of the processes r and S #
def S_i_j_k(i,j,k):
    return(np.exp(sigma_s*(pow(1-pow(rho,2),0.5))*Y[i][j] + rho * R[i][k]))

def r_i_k(i,k):
    if(R[i][k]>0):
        return(pow(R[i][k]*sigma_r,2)/4)
    else:
        return 0

# backward dynamic programming for American put option #
v=[]
v_j=[]

for j in range(0,N+1):
    v_j=[]
    for k in range(0,N+1):
        v_j+=[max(K-S_i_j_k(N,j,k),0)]
    v+=[v_j]

v=[v]
v_i=[]
v_i_j=[]
for i in range(0,N,-1):
    v_i=[]
    for j in range(0,i+1):
        v_i_j=[]
        for k in range(0, i+1):
            v_i_j=max(max((K-S_i_j_k(i,j,k)),0),np.exp(-r_i_k(i,k)*h)*(q_i_ju_ku(i,j,k)*v[0][j_u_i_j_k(i,j,k)][k_u_i_k(i,k)]+q_i_ju_kd(i,j,k)*v[0][j_u_i_j_k(i,j,k)][k_d_i_k(i,k)]+q_i_jd_ku(i,j,k)*v[0][j_d_i_j_k(i,j,k)][k_u_i_k(i,k)]+ q_i_ju_kd(i,j,k)*v[0][j_d_i_j_k(i,j,k)][k_d_i_k(i,k)]))
        v_i+=[v_i_j]
    v=[v_i]+v



