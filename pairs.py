#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 09:04:35 2019

@author: Hook
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

df1 = pd.read_csv('GSK.csv')
df2 = pd.read_csv('PFE.csv')

GSK=np.array(df1['Open'])
PFE=np.array(df2['Open'])

plt.plot(GSK,label='gsk')
plt.plot(PFE,label='pfe')
plt.legend()
plt.xlabel('day')
plt.ylabel('share price')
plt.show()

R=np.log(GSK/PFE)
V=R[1:]-R[:-1]
plt.plot(R)
plt.ylabel('log(GSK/PFE)')
plt.xlabel('day')
plt.show()

class Markov:

    """ fits a markov model to the ratio using historical data H """
    def fit(self,H,nstarts=3):
        n=H.shape[0]
        def NLL(theta):
            a,b,mu=theta
            a=np.exp(a)
            b=np.exp(b)
            Z=(H[1:]-H[:-1]-a*(mu-H[:-1]))/b
            return (n-1)*np.log(b)/2+np.sum(Z**2)/2
        
        def DNLL(theta):
            aa,bb,mu=theta
            a=np.exp(aa)
            b=np.exp(bb)
            Z=(H[1:]-H[:-1]-a*(mu-H[:-1]))/b            
            dNLL_da=np.sum(Z*-(mu-H[:-1])/b)
            dNLL_db=np.sum(Z*-(H[1:]-H[:-1]-a*(mu-H[:-1]))/b**2)+(n-1)/(2*b)
            dNLL_dmu=np.sum(Z*-a/b)
            dNLL_daa=dNLL_da*a
            dNLL_dbb=dNLL_db*b
            return np.array([dNLL_daa,dNLL_dbb,dNLL_dmu])
        
        best=np.inf
        for i in range(nstarts):
            OPT=minimize(NLL,np.random.randn(3),jac=DNLL)
            if OPT.fun<best:
                best=OPT.fun
                self.OPT=OPT
        theta=OPT.x
        self.a=np.exp(theta[0])
        self.b=np.exp(theta[1])
        self.mu=theta[2]
    
    """ samples furure trajectories from starting ratio r0 """
    def sample(self,r0,l,nsamples):
        S=np.zeros((l,nsamples))
        S[0,:]=r0
        for t in range(1,l):
            S[t,:]=S[t-1,:]+self.a*(self.mu-S[t-1,:])+self.b*np.random.randn(nsamples)
        return S

""" fit models at different time points and plot hyper-parameters """

n=R.shape[0]
MDLS=[]
updates=[200+10*t for t in range(int((n-200)/10))]
for t in updates:
    mdl=Markov()
    mdl.fit(R[t-200:t])
    MDLS.append(mdl)

A=[mdl.a for mdl in MDLS]
B=[mdl.b for mdl in MDLS]
MU=[mdl.mu for mdl in MDLS]
NLL=[mdl.OPT.fun for mdl in MDLS]

plt.plot(updates,A)
plt.xlabel('day')
plt.ylabel('a')
plt.show()

plt.plot(updates,B)
plt.xlabel('day')
plt.ylabel('b')
plt.show()

plt.plot(R,label='R')
plt.plot(updates,MU,label='mu')
plt.ylim([np.min(R),np.max(R)])
plt.xlabel('day')
plt.legend()
plt.show()

""" compare 10 day predictions to what actually happens """
""" the prediction should just be a normal so need to actually use sampling 
but its easy and i'm being lazy! """

m=len(updates)
dr=np.zeros(m)
dr_pmu=np.zeros(m)
dr_psig=np.zeros(m)

for i in range(m-1):
    mdl=MDLS[i]
    S=mdl.sample(R[updates[i]-1],11,100)
    DRS=S[-1,:]-R[updates[i]-1]
    dr_pmu[i]=np.mean(DRS)
    dr_psig[i]=np.cov(DRS)**0.5
    dr[i]=R[updates[i+1]-1]-R[updates[i]-1]    
    
plt.plot(dr_pmu,dr,'x')
plt.xlabel('expected R(t+10)-R(t)')
plt.ylabel('actual R(t+10)-R(t)')
plt.show()
print(np.corrcoef(dr_pmu,dr))

""" trading strategy bets on R going up or down"""
""" always plan ten day trades """
""" bet R goes up = long on GSK short on PFE """
""" buy £1's worth GSK and short £1's worth PFE """
""" profit = PFE_change_factor*(Ratio_change_factor-1) - fees """
""" approximate by (Ratio_change_factor-1) - fees """

""" bet G goes down = short on GSK and long on PFE  """
""" short £1's worth of GSK and buy £1's worth of PFE """
""" profit = GSK_change_factor*(1-Ratio_change_factor) - fees """
""" approximate by (1-Ratio_change_factor) - fees """

fees=0.005
expected_profit_up=np.zeros(m)
expected_profit_down=np.zeros(m)
actual_profit_up=np.zeros(m)
actual_profit_down=np.zeros(m)
for i in range(m-1):
    mdl=MDLS[i]
    S=mdl.sample(R[updates[i]-1],11,100)
    DRS=S[-1,:]-R[updates[i]-1]
    expected_profit_up[i]=np.mean(np.exp(DRS)-1-fees)
    expected_profit_down[i]=np.mean(1-np.exp(DRS)-fees)
    actual_profit_up[i]=(GSK[updates[i+1]-1]-GSK[updates[i]-1])/GSK[updates[i]-1]-(PFE[updates[i+1]-1]-PFE[updates[i]-1])/PFE[updates[i]-1]-fees
    actual_profit_down[i]=-(GSK[updates[i+1]-1]-GSK[updates[i]-1])/GSK[updates[i]-1]+(PFE[updates[i+1]-1]-PFE[updates[i]-1])/PFE[updates[i]-1]-fees

plt.plot(updates,actual_profit_up,label='actual up')
plt.plot(updates,actual_profit_down,label='actual down')
plt.plot([0,n],[0,0],color='black')
plt.xlabel('day')
plt.ylabel('profit')
plt.legend()
plt.show()

plt.plot(updates,actual_profit_up,label='actual up')
plt.plot(updates,expected_profit_up,label='expected up')
plt.plot([0,n],[0,0],color='black')
plt.xlabel('day')
plt.ylabel('profit')
plt.legend()
plt.show()

plt.plot(updates,actual_profit_down,label='actual down')
plt.plot(updates,expected_profit_down,label='expected down')
plt.plot([0,n],[0,0],color='black')
plt.xlabel('day')
plt.ylabel('profit')
plt.legend()
plt.show()

""" simulate trading strategy that bets all it has on up/down whenever it thinks that will be profitable """

P=np.ones(m)
p=1
for i in range(m-1):
    if expected_profit_up[i]>0:
        p=P[i]*(1+actual_profit_up[i])
    if expected_profit_down[i]>0:
        p=P[i]*(1+actual_profit_down[i])
    P[i+1]=p

plt.plot(P)
plt.xlabel('day')
plt.ylabel('profit')
plt.show()
    
""" plot tranding strategy profit against R """

plt.plot(updates,P,label='profit')
plt.plot(R*0.75+1.1,label='R (not to axis)')
plt.xlabel('day')
plt.legend()
plt.show()