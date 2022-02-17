#%%
"""
Created on July 05  2021
Impact of conditional expectation pricing (Black-Scholes with Jump volatility)

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import enum
import scipy.stats as st

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def GeneratePaths(NoOfPaths,NoOfSteps,S0,T,muJ,sigmaJ,r):    
    # Create empty matrices for Poisson process and for compensated Poisson process
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
                
    dt = T / float(NoOfSteps)
    X[:,0] = np.log(S0)
    S[:,0] = S0
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    J = np.random.normal(muJ,sigmaJ,[NoOfPaths])
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1]  = X[:,i] + (r - 0.5*J**2.0)*dt +J*np.sqrt(dt)* Z[:,i]
        time[i+1] = time[i] +dt
        
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S,"J":J}
    return paths

def EUOptionPriceFromMCPaths(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
    * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
    return value

def CallOption_CondExpectation(NoOfPaths,T,S0,K,J,r):
    result = np.zeros([NoOfPaths])
    
    for j in range(0,NoOfPaths):
        sigma = J[j]
        result[j] = BS_Call_Put_Option_Price(OptionType.CALL,S0,[K],sigma,0.0,T,r)
        
    return np.mean(result)

def mainCalculation():
    NoOfPaths = 25
    NoOfSteps = 500
    T = 5
    muJ = 0.3
    sigmaJ = 0.005
    
    S0 =100
    r  =0.00
    Paths = GeneratePaths(NoOfPaths,NoOfSteps,S0, T,muJ,sigmaJ,r)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
           
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.show()
    
    # Check the convergence for a given strike
    K = 80
    CP =OptionType.CALL
    
    NGrid = range(100,10000,1000)
    NoOfRuns = len(NGrid)
    
    resultMC = np.zeros([NoOfRuns])
    resultCondExp = np.zeros([NoOfRuns])
       
    for (i,N) in enumerate(NGrid):
            print(N)
            Paths = GeneratePaths(N,NoOfSteps,S0, T,muJ,sigmaJ,r)
            timeGrid = Paths["time"]
            S = Paths["S"]
            resultMC[i] = EUOptionPriceFromMCPaths(CP,S[:,-1],K,T,r)
            
            J = Paths["J"]

            resultCondExp[i]=CallOption_CondExpectation(N,T,S0,K,J,r)
    
    plt.figure(3)
    plt.plot(NGrid,resultMC)
    plt.plot(NGrid,resultCondExp)
    plt.legend(['MC','Conditional Expectation'])
    plt.title('Call Option Price- Convergence')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option price for a given strike, K')
    plt.grid()
    plt.show()
                       
mainCalculation()
