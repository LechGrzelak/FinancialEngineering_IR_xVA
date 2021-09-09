#%%
"""
Created on July 05  2021
Incentive function as a function of a swap rate or the differential w.r.t. "old" mortgage rate

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak and Emanuele Cassamassima
"""
import numpy as np
import matplotlib.pyplot as plt

def Annuity(rate,notional,periods,CPR):
    # it returns a matrix M such that
    # M = [t  notional(t)  prepayment(t)  notional_quote(t)  interest_quote(t)  installment(t)]
    # WARNING! here "rate" and "periods" are quite general, the choice of getting year/month/day.. steps, depends on the rate
    # that the function receives. So, it is necessary to pass the correct rate to the function
    M = np.zeros((periods + 1,6))
    M[:,0] = np.arange(periods + 1) # we define the times
    M[0,1] = notional
    for t in range(1,periods + 1):
        # we are computing the installment at time t knowing the oustanding at time (t-1)
        remaining_periods = periods - (t - 1)  
        
        # Installment, C(t_i) 
        M[t,5] = rate * M[t-1,1]/(1 - 1/(1 + rate)**remaining_periods) 
        
        # Interest rate payment, I(t_i) = r * N(t_{i})
        M[t,4] = rate * M[t-1,1] 
        
        # Notional payment, Q(t_i) = C(t_i) - I(t_i)
        M[t,3] = M[t,5] - M[t,4] 
        
        # Prepayment, P(t_i)= Lambda * (N(t_i) -Q(t_i))
        M[t,2] = CPR * (M[t-1,1] - M[t,3]) 
        
        # notional, N(t_{i+1}) = N(t_{i}) - lambda * (Q(t_{i} + P(t_i)))
        M[t,1] = M[t-1,1] - M[t,3] - M[t,2] 
    return M

def mainCode():


    IncentiveFunction = lambda x : 0.04 + 0.1/(1 + np.exp(115 * (0.02-x))) 
    
    oldRate = 0.05
    newRate = np.linspace(-0.05,0.15,25)
    
    epsilon = oldRate-newRate
    incentive = IncentiveFunction(epsilon)
    
    plt.figure(1)
    plt.plot(newRate,incentive)
    plt.xlabel('S(t)')
    plt.ylabel('Incentive')
    plt.grid()
    
    plt.figure(2)
    plt.plot(epsilon,incentive)
    plt.xlabel('epsilon= K - S(t)')
    plt.ylabel('Incentive')
    plt.grid()

    return 0.0

mainCode()
    
    