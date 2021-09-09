#%%
"""
Created on August 25  2021
Value-At-Risk computation based on Monte Carlo simulation of the Hull-White model

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""

import numpy as np
import enum 
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import scipy.optimize as optimize

# This class defines puts and calls
class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    #theta = lambda t: 0.1 +t -t
    #print("changed theta")
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    R[:,0]=r0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Outputs
    paths = {"time":time,"R":R}
    return paths

def HW_theta(lambd,eta,P0T):
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))
    #print("CHANGED THETA")
    return theta#lambda t: 0.1+t-t
    
def HW_A(lambd,eta,P0T,T1,T2):
    tau = T2-T1
    zGrid = np.linspace(0.0,tau,250)
    B_r = lambda tau: 1.0/lambd * (np.exp(-lambd *tau)-1.0)
    theta = HW_theta(lambd,eta,P0T)    
    temp1 = lambd * integrate.trapz(theta(T2-zGrid)*B_r(zGrid),zGrid)
    
    temp2 = eta*eta/(4.0*np.power(lambd,3.0)) * (np.exp(-2.0*lambd*tau)*(4*np.exp(lambd*tau)-1.0) -3.0) + eta*eta*tau/(2.0*lambd*lambd)
    
    return temp1 + temp2

def HW_B(lambd,eta,T1,T2):
    return 1.0/lambd *(np.exp(-lambd*(T2-T1))-1.0)

def HW_ZCB(lambd,eta,P0T,T1,T2,rT1):
    n = np.size(rT1) 
        
    if T1<T2:
        B_r = HW_B(lambd,eta,T1,T2)
        A_r = HW_A(lambd,eta,P0T,T1,T2)
        return np.exp(A_r + B_r *rT1)
    else:
        return np.ones([n])


def HWMean_r(P0T,lambd,eta,T):
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,2500)
    temp =lambda z: theta(z) * np.exp(-lambd*(T-z))
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    return r_mean

def HW_r_0(P0T,lambd,eta):
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0

def HW_Mu_FrwdMeasure(P0T,lambd,eta,T):
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,500)
    
    theta_hat =lambda t,T:  theta(t) + eta*eta / lambd *1.0/lambd * (np.exp(-lambd*(T-t))-1.0)
    
    temp =lambda z: theta_hat(z,T) * np.exp(-lambd*(T-z))
    
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    
    return r_mean

def HWVar_r(lambd,eta,T):
    return eta*eta/(2.0*lambd) *( 1.0-np.exp(-2.0*lambd *T))

def HWDensity(P0T,lambd,eta,T):
    r_mean = HWMean_r(P0T,lambd,eta,T)
    r_var = HWVar_r(lambd,eta,T)
    return lambda x: st.norm.pdf(x,r_mean,np.sqrt(r_var))

def HW_SwapPrice(CP,notional,K,t,Ti,Tm,n,r_t,P0T,lambd,eta):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t

    if n == 1:
        ti_grid =np.array([Ti,Tm])
    else:
        ti_grid = np.linspace(Ti,Tm,n)
    tau = ti_grid[1]- ti_grid[0]
    
    # overwrite Ti if t>Ti
    prevTi = ti_grid[np.where(ti_grid<t)]
    if np.size(prevTi) > 0: #prevTi != []:
        Ti = prevTi[-1]
    
    # Now we need to handle the case when some payments are already done
    ti_grid = ti_grid[np.where(ti_grid>t)]          

    temp= np.zeros(np.size(r_t));
    
    P_t_TiLambda = lambda Ti : HW_ZCB(lambd,eta,P0T,t,Ti,r_t)
    
    for (idx,ti) in enumerate(ti_grid):
        if ti>Ti:
            temp = temp + tau * P_t_TiLambda(ti)
            
    P_t_Ti = P_t_TiLambda(Ti)
    P_t_Tm = P_t_TiLambda(Tm)
    
    if CP==OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP==OptionTypeSwap.RECEIVER:
        swap = K * temp- (P_t_Ti - P_t_Tm)
    
    return swap*notional

def Portfolio(P0T,r_t,lambd,eta):
    
            #IRSwap(CP,           notional,          K,   t,   Ti,  Tm,   n,      P0T):
    value = HW_SwapPrice(OptionTypeSwap.RECEIVER,1000000,0.02,0.0, 0.0,  20,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.PAYER, 500000,   0.01,0.0, 0.0,  10,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.RECEIVER,25000,0.02,0.0, 0.0,  30,  60,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.PAYER,74000,0.005,0.0, 0.0,  5,  10,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.RECEIVER,254000,0.032,0.0, 0.0,  15,  10,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.RECEIVER,854000,0.01,0.0, 0.0,  7,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.PAYER,900000,0.045,0.0, 0.0,  10,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.PAYER,400000,0.02,0.0, 0.0,  10,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.RECEIVER,1000000,0.01,0.0, 0.0,  14,  20,r_t,  P0T,lambd,eta) +\
            HW_SwapPrice(OptionTypeSwap.PAYER,115000,0.06,0.0, 0.0,  9,  10,r_t,  P0T,lambd,eta) 
            #HW_SwapPrice(CP,notional,K,t,Ti,Tm,n,r_t,P0T,lambd,eta)
    return value

def mainCalculation():
    NoOfPaths = 2000
    NoOfSteps = 100
    lambd     = 0.5
    eta       = 0.03
    
    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.001*T)
    r0 = HW_r_0(P0T,lambd,eta)
    
    # In this experiment we compare ZCB from the Market and Analytical expression
    N = 25
    T_end = 50
    Tgrid= np.linspace(0,T_end,N)
    
    Exact = np.zeros([N,1])
    Proxy= np.zeros ([N,1])
    for i,Ti in enumerate(Tgrid):
        Proxy[i] = HW_ZCB(lambd,eta,P0T,0.0,Ti,r0)
        Exact[i] = P0T(Ti)
        
    plt.figure(1)
    plt.grid()
    plt.plot(Tgrid,Exact,'-k')
    plt.plot(Tgrid,Proxy,'--r')
    plt.legend(["Analytcal ZCB","Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')
    
    
    # Here we simulate the exposure profiles for a swap, using the HW model    
    T_end = 20    
    paths= GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T_end ,P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1]-timeGrid[0]
    
    # Here we compare the price of an option on a ZCB from Monte Carlo and Analytical expression    
    M_t = np.zeros([NoOfPaths,NoOfSteps])
            
    for i in range(0,NoOfPaths):
        M_t[i,:] = np.exp(np.cumsum(r[i,0:-1])*dt)
        
    # Compute exposure profile
    r0 = r[0,0]
    
    stepSize = 10
    V_M = np.zeros([NoOfPaths,NoOfSteps-stepSize])
    
    for i in range(0,NoOfSteps-stepSize):
        dr = r[:,i + stepSize] - r[:,i]
        V_t0 = Portfolio(P0T, r[:,0] + dr,lambd,eta)
        V_M[:,i] = V_t0
    
    plt.figure(2)
    V_t0_vec = np.matrix.flatten(V_M)
    plt.hist(V_t0_vec,100)
    plt.grid()
    
    print('Value V(t_0)= ',Portfolio(P0T,r[0,0],lambd,eta))
        
    # VaR calculation
    alpha = 0.05
    HVaR_estimate = np.quantile(V_t0_vec,alpha)
    print('(H)VaR for alpha = ', alpha, ' is equal to=', HVaR_estimate)
    
    # Expected shortfal
    condLosses = V_t0_vec[V_t0_vec < HVaR_estimate]
    print('P&L which < VaR_alpha =',condLosses)
    ES = np.mean(condLosses)
    
    print('Expected shortfal = ', ES)
    
    plt.plot(HVaR_estimate,0,'or')
    plt.plot(ES,0,'ok')
    plt.legend(['VaR','ES','P&L'])
    
    
mainCalculation()