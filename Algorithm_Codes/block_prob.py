import numpy as np
import scipy.integrate as integrate

# This file calculates the blocking probability for general N, N_1 and N_2 which is used in the 3-state Linear-alpha algorithm in dispatch_main.py

def ErlangLoss(Lambda, Mu, N = None): # take it out once figured out what was the problem for the error. This is now a copy of the function in dispatch_main
    """
    The Erlangloss Model is exactly the same as MMN0 and this returns the whole probability distribution.
    Solves the Erlang loss system
    :param Lambda:
    :param Mu:
    :param N:
    :return: Probability distribution P_n
    """
    if N == 0: # if there is 0 unit. The block probability is 1
        return [1]
    if N is not None: # If there is a size N, we constitute the Lambda and Mu vectors manually 
        Lambda = np.ones(N)*Lambda
        Mu = Mu*(np.arange(N)+1)
    else: # if the Lambdas and Mus are already given in vector form then no need to do anything
        N = len(Lambda)
    LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
    Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
    P_n = Prod/sum(Prod)
    return P_n

def Hayward_Approx_P_b(n, r):
    # The Hayward Approx function that is a genralized form of the Erlang loss formula that allows continuous number of units
    B = 1 / (r**(-n) * np.exp(r) * integrate.quad(lambda x: np.exp(-x) * x**n, r, np.inf)[0])
    return B

def peakedness_z(N, a, b):
    # The peakedness of the arrival process
    return 1 - b + a/(N - a + b + 1)

def P_b1(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N):
    # This assumes that all separate units are dispatched before joint units, and obtains the approximate blocking probability of this system
    # This value is close to P_b2
    N_c = N - N_1 - N_2
    a_1, a_2 = Lambda_1/Mu_1, Lambda_2/Mu_2 # offered load
    P_b1_prim_1, P_b1_prim_2 = ErlangLoss(Lambda_1, Mu_1, N=N_1)[-1], ErlangLoss(Lambda_2, Mu_2, N=N_2)[-1]
    b_1, b_2 = a_1*P_b1_prim_1, a_2*P_b1_prim_2
    v_1, v_2 = b_1 * peakedness_z(N_1, a_1, b_1), b_2 * peakedness_z(N_2, a_2, b_2)
    z = (v_1 + v_2)/(b_1 + b_2)
    P_b1_1 = P_b1_prim_1*Hayward_Approx_P_b(N_c/z, (b_1 + b_2)/z) 
    P_b1_2 = P_b1_prim_2*Hayward_Approx_P_b(N_c/z, (b_1 + b_2)/z)
    return P_b1_1, P_b1_2

def P_b2(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N):
    # This assumes that all joint units are dispatched before separate units, and obtains the approximate blocking probability of this system
    # This value is close to P_b1
    N_c = N - N_1 - N_2
    a_1, a_2 = Lambda_1/Mu_1, Lambda_2/Mu_2 # offered load
    if N_c > 0:
        P_b2_prim = ErlangLoss(Lambda_1+Lambda_2, (Lambda_1+Lambda_2)/(Lambda_1/Mu_1+Lambda_2/Mu_2), N_c)[-1] # blocking probability
        b_1, b_2 = a_1*P_b2_prim, a_2*P_b2_prim
        z_1, z_2 = peakedness_z(N_c, a_1+a_2, b_1+b_2), peakedness_z(N_c, a_1+a_2, b_1+b_2) # this don't need to get v as in P_b1
    else:
        b_1, b_2 = a_1, a_2
        z_1, z_2 = 1, 1
    
    P_b2_1 = b_1/a_1 * Hayward_Approx_P_b(N_1/z_1, b_1/z_1) 
    P_b2_2 = b_2/a_2 * Hayward_Approx_P_b(N_2/z_2, b_2/z_2)
    return P_b2_1, P_b2_2