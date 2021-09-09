#%%
"""
Created on July 05  2021
Annuity mortgage- payment profile

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak and Emanuele Casamassima
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
        
        # Interest rate payment, I(t_i) = K * N(t_{i})
        M[t,4] = rate * M[t-1,1] 
        
        # Notional payment, Q(t_i) = C(t_i) - I(t_i)
        M[t,3] = M[t,5] - M[t,4] 
        
        # Prepayment, P(t_i)= Lambda * (N(t_i) -Q(t_i))
        M[t,2] = CPR * (M[t-1,1] - M[t,3]) 
        
        # notional, N(t_{i+1}) = N(t_{i}) - lambda * (Q(t_{i} + P(t_i)))
        M[t,1] = M[t-1,1] - M[t,3] - M[t,2] 
    return M

def mainCode():

    # Initial notional
    N0     = 1000000
    
    # Interest rates from a bank
    r = 0.05
    
    # Prepayment rate, 0.1 = 10%
    Lambda = 0.1

    # For simplicity we assume 1 as unit (yearly payments of mortgage)
    T_end = 30
    M = Annuity(r,N0,T_end,Lambda)
    
    for i in range(0,T_end+1):
        print("Ti={0}, Notional={1:.0f}, Prepayment={2:.0f}, Notional Repayment={3:.0f}, Interest Rate={4:.0f}, Installment={5:.0f} ".format(M[i,0],M[i,1],M[i,2],M[i,3],M[i,4],M[i,5]))
    
    plt.figure(1)
    plt.plot(M[:,0],M[:,1],'.r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('notional')
    return 0.0

mainCode()
    
    