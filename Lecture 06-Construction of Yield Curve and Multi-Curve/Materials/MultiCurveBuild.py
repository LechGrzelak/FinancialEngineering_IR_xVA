#%%
"""
Created on June 27 2021
Construction of a multi- curve for a given set of swap instruments

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d

# This class defines puts and calls
class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0
    
def IRSwap(CP,notional,K,t,Ti,Tm,n,P0T):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t
    ti_grid = np.linspace(Ti,Tm,int(n))
    tau = ti_grid[1]- ti_grid[0]
    
    temp= 0.0
        
    for (idx,ti) in enumerate(ti_grid):
        if idx>0:
            temp = temp + tau * P0T(ti)
            
    P_t_Ti = P0T(Ti)
    P_t_Tm = P0T(Tm)
    
    if CP==OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP==OptionTypeSwap.RECEIVER:
        swap = K * temp - (P_t_Ti - P_t_Tm)
    
    return swap * notional


def IRSwapMultiCurve(CP,notional,K,t,Ti,Tm,n,P0T,P0TFrd):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t
    ti_grid = np.linspace(Ti,Tm,int(n))
    tau = ti_grid[1]- ti_grid[0]
    
    swap = 0.0
        
    for (idx,ti) in enumerate(ti_grid):
        #L(t_0,t_{k-1},t_{k}) from the forward curve
        if idx>0:
            L_frwd = 1.0/tau * (P0TFrd(ti_grid[idx-1])-P0TFrd(ti_grid[idx])) / P0TFrd(ti_grid[idx])
            swap = swap + tau * P0T(ti_grid[idx]) * (L_frwd - K)
            
    return swap * notional


def P0TModel(t,ti,ri,method):
    rInterp = method(ti,ri)
    r = rInterp(t)
    return np.exp(-r*t)

def YieldCurve(instruments, maturities, r0, method, tol):
    r0 = deepcopy(r0)
    ri = MultivariateNewtonRaphson(r0, maturities, instruments, method, tol=tol)
    return ri

def MultivariateNewtonRaphson(ri, ti, instruments, method, tol):
    err = 10e10
    idx = 0
    while err > tol:
        idx = idx +1
        values = EvaluateInstruments(ti,ri,instruments,method)
        J = Jacobian(ti,ri, instruments, method)
        J_inv = np.linalg.inv(J)
        err = - np.dot(J_inv, values)
        ri[0:] = ri[0:] + err 
        err = np.linalg.norm(err)
        print('index in the loop is',idx,' Error is ', err)
    return ri

def Jacobian(ti, ri, instruments, method):
    eps = 1e-05
    swap_num = len(ti)
    J = np.zeros([swap_num, swap_num])
    val = EvaluateInstruments(ti,ri,instruments,method)
    ri_up = deepcopy(ri)
    
    for j in range(0, len(ri)):
        ri_up[j] = ri[j] + eps  
        val_up = EvaluateInstruments(ti,ri_up,instruments,method)
        ri_up[j] = ri[j]
        dv = (val_up - val) / eps
        J[:, j] = dv[:]
    return J

def EvaluateInstruments(ti,ri,instruments,method):
    P0Ttemp = lambda t: P0TModel(t,ti,ri,method)
    val = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        val[i] = instruments[i](P0Ttemp)
    return val

def linear_interpolation(ti,ri):
    interpolator = lambda t: np.interp(t, ti, ri)
    return interpolator

def spline_interpolate(ti,ri):
    interpolator = splrep(ti, ri, s=0.01)
    interp = lambda t: splev(t,interpolator)
    return interp

def scipy_1d_interpolate(ti, ri):
    interpolator = lambda t: interp1d(ti, ri, kind='quadratic')(t)
    return interpolator

#def ComputeDelta(instrument,)

def mainCode():
    
    # Convergence tolerance
    tol = 1.0e-15
    # Initial guess for the spine points
    r0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])   
    # Interpolation method
    method = linear_interpolation
    
    # Construct swaps that are used for building of the yield curve
    K   = np.array([0.04/100.0,	0.16/100.0,	0.31/100.0,	0.81/100.0,	1.28/100.0,	1.62/100.0,	2.22/100.0,	2.30/100.0])
    mat = np.array([1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0])
    
    #                   IRSwap(CP,           notional,K,   t,   Ti,Tm,   n,P0T)
    swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[0],0.0,0.0,mat[0],4*mat[0],P0T)
    swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[1],0.0,0.0,mat[1],5*mat[1],P0T)
    swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[2],0.0,0.0,mat[2],6*mat[2],P0T)
    swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[3],0.0,0.0,mat[3],7*mat[3],P0T)
    swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[4],0.0,0.0,mat[4],8*mat[4],P0T)
    swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[5],0.0,0.0,mat[5],9*mat[5],P0T)
    swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[6],0.0,0.0,mat[6],10*mat[6],P0T)
    swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,K[7],0.0,0.0,mat[7],11*mat[7],P0T)
    instruments = [swap1,swap2,swap3,swap4,swap5,swap6,swap7,swap8]
    
    # determine optimal spine points
    ri = YieldCurve(instruments, mat, r0, method, tol)
    print('\n Spine points are',ri,'\n')
    
    # Build a ZCB-curve/yield curve from the spine points
    P0T_Initial = lambda t: P0TModel(t,mat,r0,method)
    P0T = lambda t: P0TModel(t,mat,ri,method)
    
    # price back the swaps
    swapsModel = np.zeros(len(instruments))
    swapsInitial = np.zeros(len(instruments))
    for i in range(0,len(instruments)):
        swapsModel[i] = instruments[i](P0T)
        swapsInitial[i] = instruments[i](P0T_Initial)
    
    print('Prices for Pas Swaps (initial) = ',swapsInitial,'\n')
    print('Prices for Par Swaps = ',swapsModel,'\n')
    
    
    # Multi Curve extension- simple sanity check
    P0TFrd = deepcopy(P0T)
    Ktest = 0.2
    swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER,1,Ktest,0.0,0.0,mat[0],4*mat[0],P0T)
    swap1MC = lambda P0T: IRSwapMultiCurve(OptionTypeSwap.PAYER,1,Ktest,0.0,0.0,mat[0],4*mat[0],P0T,P0TFrd)
    print('Sanity check: swap1 = {0}, swap2 = {1}'.format(swap1(P0T),swap1MC(P0T)))
    
    # Forward curve settings
    r0Frwd = np.array([0.01, 0.01, 0.01, 0.01])   
    KFrwd   = np.array([0.09/100.0,	0.26/100.0,	0.37/100.0,	1.91/100.0])
    matFrwd = np.array([1.0, 2.0, 3.0, 5.0])
    
    # At this point we already know P(0,T) for the discount curve
    P0TDiscount = lambda t: P0TModel(t,mat,ri,method)
    swap1Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER,1,KFrwd[0],0.0,0.0,matFrwd[0],4*matFrwd[0],P0TDiscount,P0TFrwd)
    swap2Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER,1,KFrwd[1],0.0,0.0,matFrwd[1],5*matFrwd[1],P0TDiscount,P0TFrwd)
    swap3Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER,1,KFrwd[2],0.0,0.0,matFrwd[2],6*matFrwd[2],P0TDiscount,P0TFrwd)
    swap4Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER,1,KFrwd[3],0.0,0.0,matFrwd[3],7*matFrwd[3],P0TDiscount,P0TFrwd)
    
    instrumentsFrwd = [swap1Frwd,swap2Frwd,swap3Frwd,swap4Frwd]
    
    # determine optimal spine points for the forward curve
    riFrwd = YieldCurve(instrumentsFrwd, matFrwd, r0Frwd, method, tol)
    print('\n Frwd Spine points are',riFrwd,'\n')
    
    # Build a ZCB-curve/yield curve from the spine points
    P0TFrwd_Initial = lambda t: P0TModel(t,matFrwd,r0Frwd,method)
    P0TFrwd         = lambda t: P0TModel(t,matFrwd,riFrwd,method)
    # price back the swaps
    swapsModelFrwd   = np.zeros(len(instrumentsFrwd))
    swapsInitialFrwd = np.zeros(len(instrumentsFrwd))
    
    for i in range(0,len(instrumentsFrwd)):
        swapsModelFrwd[i]   = instrumentsFrwd[i](P0TFrwd)
        swapsInitialFrwd[i] = instrumentsFrwd[i](P0TFrwd_Initial)
    
    print('Prices for Pas Swaps (initial) = ',swapsInitialFrwd,'\n')
    print('Prices for Par Swaps = ',swapsModelFrwd,'\n')
    
    print(swap1Frwd(P0TFrwd))
    
    t = np.linspace(0,10,100)
    plt.figure()
    plt.plot(t,P0TDiscount(t),'--r')
    plt.plot(t,P0TFrwd(t),'-b')
    plt.legend(['discount','forecast'])
    
    return 0.0

mainCode()
    
    