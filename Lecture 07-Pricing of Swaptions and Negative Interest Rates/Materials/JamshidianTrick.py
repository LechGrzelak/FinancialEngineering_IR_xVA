#%%
"""
Created on July 18 2021
Jamshidian's trick for handling E max sum -> sum E max 

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def PsiSum(psi,N,x):
    temp = 0
    for i in range(0,N):
        temp = temp + psi(i,x)
    return temp

def JamshidianTrick(psi,N,K):
    A = lambda x: PsiSum(psi,N,x) - K
    result = optimize.newton(A,0.1)
    return result

def Main():
    NoOfSamples = 1000
    X =  np.random.normal(0.0,1.0,[NoOfSamples,1])
    psi_i = lambda i,X: np.exp(-i*np.abs(X))
    
    # Number of terms
    N = 15
    
    A = 0
    for i in range(0,N):
        A = A + psi_i(i,X)
    
    K = np.linspace(2,10,10)
    resultMC = np.zeros(len(K))
    for (i,Ki) in enumerate(K):
        resultMC[i] = np.mean(np.maximum(A-Ki,0))
    
    # Jamshidians trick
    resultJams = np.zeros(len(K))
    for i,Ki in enumerate(K):
        # Compute optimal K*
        optX = JamshidianTrick(psi_i,N,Ki)
        A = 0
        for j in range(0,N):
            A = A + np.mean(np.maximum(psi_i(j,X)-psi_i(j,optX),0))
        resultJams[i] = A
        
    plt.figure()
    plt.plot(K,resultMC)
    plt.plot(K,resultJams,'--r')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('expectation')
    plt.legend(['Monte Carlo','Jamshidians trick'])
    
Main()