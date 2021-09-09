#%%
"""
Created on July 05  2021
Stochastic amortization given the incentive function and irrational/rational behavior profile

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak and Emanuele Cassamassima
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as integrate

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
    return theta

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

def SwapRateHW(t,Ti,Tm,n,r_t,P0T,lambd,eta):
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

    swapRate = (P_t_Ti - P_t_Tm) / temp
    
    return swapRate

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
        M[t,3] = 0                  # notional quote, 0 for bullet mortgage
        scheduled_oustanding = M[t-1,1] - M[t,3]
        M[t,2] = scheduled_oustanding * CPR[t]    # prepayment
        M[t,1] = scheduled_oustanding - M[t,2] # notional(t) = notional(t-1) - (notional quote + prepayment)
        M[t,5] = M[t,4] + M[t,2] + M[t,3]
        
    M[periods,4] = rate*M[periods-1,1] # interest quote
    M[periods,3] = M[periods-1,1]      # notional quote
    M[periods,5] = M[periods,4] + M[periods,2] + M[periods,3]
    return M    

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
        M[t,2] = CPR[t] * (M[t-1,1] - M[t,3]) 
        
        # notional, N(t_{i+1}) = N(t_{i}) - lambda * (Q(t_{i} + P(t_i)))
        M[t,1] = M[t-1,1] - M[t,3] - M[t,2] 
    return M

def mainCode():

    Irrational = lambda x : 0.04 + 0.1/(1 + np.exp(200 * (-x))) 
    Rational   = lambda x : 0.04*(x>0.0)

    
    IncentiveFunction = Irrational
    
    K = 0.05
    newRate = np.linspace(-0.1,0.1,150)
    
    epsilon = K - newRate
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

    # Stochastic interest rates
    NoOfPaths = 2000
    NoOfSteps = 30
    lambd     = 0.05
    eta       = 0.01
    
    # End date of the underlying swap / mortgage
    Tend = 30 

    # Market ZCB
    P0T = lambda T: np.exp(-0.05*T)
    paths =  GeneratePathsHWEuler(NoOfPaths, NoOfSteps,Tend, P0T, lambd, eta)
    R = paths["R"]
    tiGrid = paths["time"]

    # Compute swap rates, at this point we assume that the incentive is driven by the CMS rate
    S = np.zeros([NoOfPaths,NoOfSteps+1])
    for (i,ti) in enumerate(tiGrid):
        S[:,i] = SwapRateHW(ti,ti,Tend+ti,30,R[:,i],P0T,lambd,eta)
    
    # Incentive for the new swap rate
    epsilon = K - S[:,-1]
    incentive = IncentiveFunction(epsilon)
    plt.figure(3)
    plt.plot(epsilon,incentive,'.r')
    plt.xlabel('epsilon= K - S(t)')
    plt.ylabel('Incentive')
    plt.grid()
    plt.title('Incentive for prepayment given stochastis S(t)')

    plt.figure(4)
    plt.hist(S[:,-1],bins=50)
    plt.grid()
    plt.title('Swap distribution at Tend')

    # Building up of stochastic notional N(ti) for every ti
    MortgageProfile =  Annuity
    notional = 1000000
    N = np.zeros([NoOfPaths,NoOfSteps+1])
    
    for i in range(0,NoOfPaths):
        epsilon =  K - S[i,:]
        Lambda = IncentiveFunction(epsilon)
        NotionalProfile = MortgageProfile(K,notional,NoOfSteps,Lambda)
        N[i,:] = NotionalProfile[:,1]

    plt.figure(6)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('notional')
    
    n = 100
    for k in range(0,n):
        plt.plot(tiGrid,N[k,:],'-b')
    
    AnnuityProfile_NoPrepayment = MortgageProfile(K,notional,NoOfSteps,np.zeros(NoOfSteps+1))
    plt.plot(tiGrid,AnnuityProfile_NoPrepayment[:,1],'--r')
    plt.title('Notional profile')
        
    return 0.0

mainCode()
    
    