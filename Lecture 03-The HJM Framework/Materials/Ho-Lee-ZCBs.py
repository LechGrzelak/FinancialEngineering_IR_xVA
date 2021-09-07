#%%
"""
Created on July 12 2021
Ho-Lee Model, Simulation of the Model + Computation of ZCBs, P(0,t)

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""

import numpy as np
import matplotlib.pyplot as plt

def f0T(t,P0T):
    # time-step needed for differentiation
    dt = 0.01    
    expr = - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    return expr

def GeneratePathsHoLeeEuler(NoOfPaths,NoOfSteps,T,P0T,sigma):    
    
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.01,P0T)
    theta = lambda t: (f0T(t+dt,P0T)-f0T(t-dt,P0T))/(2.0*dt) + sigma**2.0*t 
     
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    M = np.zeros([NoOfPaths, NoOfSteps+1])
    M[:,0]= 1.0
    R[:,0]=r0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + theta(time[i]) * dt + sigma* (W[:,i+1]-W[:,i])
        M[:,i+1] = M[:,i] * np.exp((R[:,i+1]+R[:,i])*0.5*dt)
        time[i+1] = time[i] +dt
        
    # Outputs
    paths = {"time":time,"R":R,"M":M}
    return paths

def mainCalculation():
    NoOfPaths = 25000
    NoOfSteps = 500
       
    sigma = 0.007
        
    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.1*T)
       
    # In this experiment we compare ZCB from the Market and Monte Carlo
    "Monte Carlo part"   
    T = 40
    paths= GeneratePathsHoLeeEuler(NoOfPaths,NoOfSteps,T,P0T,sigma)
    M = paths["M"]
    ti = paths["time"]
        
    # Here we compare the price of an option on a ZCB from Monte Carlo and Analytical expression    
    P_t = np.zeros([NoOfSteps+1])
    for i in range(0,NoOfSteps+1):
        P_t[i] = np.mean(1.0/M[:,i])
   
    plt.figure(1)
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('P(0,T)')
    plt.plot(ti,P0T(ti))
    plt.plot(ti,P_t,'--r')
    plt.legend(['P(0,t) market','P(0,t) Monte Carlo'])
    plt.title('ZCBs from Ho-Lee Model')
    
mainCalculation()








