#%%
"""
Created on July 05  2021
Bullet mortgage- payment profile

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak and Emanuele Casamassima
"""
import numpy as np
import matplotlib.pyplot as plt

def Bullet(rate,notional,periods,CPR):
    # it returns a matrix M such that
    # M = [t  notional(t)  prepayment(t)  notional_quote(t)  interest_(t)  installment(t)]
    # WARNING! here "rate" and "periods" are quite general, the choice of getting year/month/day.. steps, depends on the rate
    # that the function receives. So, it is necessary to pass the correct rate to the function
    M = np.zeros((periods + 1,6))
    M[:,0] = np.arange(periods + 1) # we define the times
    M[0,1] = notional
    for t in range(1,periods):
        M[t,4] = rate*M[t-1,1]      # interest quote
        M[t,3] = 0                  # repayment, 0 for bullet mortgage
        scheduled_oustanding = M[t-1,1] - M[t,3]
        M[t,2] = scheduled_oustanding * CPR    # prepayment
        M[t,1] = scheduled_oustanding - M[t,2] # notional(t) = notional(t-1) - (repayment + prepayment)
        M[t,5] = M[t,4] + M[t,2] + M[t,3]
        
    M[periods,4] = rate*M[periods-1,1] # interest quote
    M[periods,3] = M[periods-1,1]      # notional quote
    M[periods,5] = M[periods,4] + M[periods,2] + M[periods,3]
    return M

def mainCode():

    # Initial notional
    N0     = 1000000
    
    # Interest rates from a bank
    r = 0.05
    
    # Prepayment rate, 0.1 = 10%
    Lambda = 0.01

    # For simplicity we assume 1 as unit (yearly payments of mortgage)
    T_end = 30
    M = Bullet(r,N0,T_end,Lambda)
    
    for i in range(0,T_end+1):
        print("Ti={0}, Notional={1:.0f}, Prepayment={2:.0f}, Notional Repayment={3:.0f}, Interest Rate={4:.0f}, Installment={5:.0f} ".format(M[i,0],M[i,1],M[i,2],M[i,3],M[i,4],M[i,5]))
    
    plt.figure(1)
    plt.plot(M[:,0],M[:,1],'-r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('notional')
    
    return 0.0

mainCode()
    
    