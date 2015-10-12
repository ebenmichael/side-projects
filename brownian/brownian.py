# -*- coding: utf-8 -*-
"""
Misc functions to simulate brownian motion
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import scipy.stats as stat

#simulate n steps of size dt of standard brownian motion
def brownian(t,dt):
    return(np.cumsum(np.random.normal(0,sqrt(dt),t/dt)))
#simulate n steps of size dt of general bronwian motion with drift m and dispersion v
def genBrownian(init,m,v,t,dt):
    return(init + np.cumsum(np.random.normal(m*dt,v*sqrt(dt),t/dt)))

#simulate n steps of size dt of geometric brownian motion with relative drift m, 
#and relative dispersion v
#e^(mt + vB(t))
def geoBrownian(init,m,v,t,dt):
    return(init*np.exp(genBrownian(0,m,v,t,dt)))
        
        

#plot multiple simulations 
def plotSim1d(title,xlab,num,sim,**kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = kwargs["t"]
    dt = kwargs["dt"]
    ts = np.linspace(0,t-1,t/dt)
    for i in range(0,num):
        ax.plot(ts,sim(**kwargs),alpha = .4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='x',which='both',top='off')
    plt.tick_params(axis='y',which='both',right='off')
    plt.grid(linestyle='-', color = 'black', alpha = .5)
    ax.xaxis.grid(False)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Value")
    plt.show()
    
#simulate k-dim'l standard brownian motion 
def kDimBrownian(k,t,dt):
    return(np.cumsum(np.random.multivariate_normal(np.zeros(k),dt * np.identity(k),int(t/dt)).T,axis = 1))

#simulate k-dim'l generalized brownian motion with drift m and dispersion matrix v
def kDimGenBrownian(k,init,m,cov,t,dt):
    cov = np.array(cov)
    init = np.array(init)
    m = np.array(m)
    return(np.sum(init,np.cumsum(np.random.multivariate_normal(m,dt * cov,int(t/dt)))))
#plots path of simulation from 0 to t-1
def plotSimkd(num,sim,**kwargs):
    fig = plt.figure()
    k = kwargs["k"]
    if k == 3:
        ax = fig.add_subplot(111,projection = "3d")
    else:
        ax = fig.add_subplot(111)
    t = kwargs["t"]
    dt = kwargs["dt"]
    ts = ts = np.linspace(0,t-1,t/dt)
    for i in range(0,num):
        sample = sim(**kwargs)
        if k == 3:
            ax.plot(sample[0],sample[1],sample[2],alpha = .4)
        else:
            ax.plot(sample[0],sample[1],alpha = .4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='x',which='both',top='off')
    plt.tick_params(axis='y',which='both',right='off')
    plt.grid(linestyle='-', color = 'black', alpha = .5)
    ax.xaxis.grid(True)
    plt.show()
    
    
#Euler method to simulate SDE given drift function mu(X,t) and dispersion 
#function sigma(X,t)
    
def Euler1d(init,mu,sigma,t,dt,*muargs,**sigargs):
    z = np.random.standard_normal(t/dt)
    x = np.zeros(t/dt)
    x[0] = init
    t_curr = 0
    for i in range(1,int(t/dt)):
        x[i] = x[i-1] + mu(x[i-1],t_curr,*muargs)*dt +\
                sigma(x[i-1],t,**sigargs)*sqrt(dt)*z[i-1]
        t_curr += dt
    return(x)   
#mu function for OU process with speed of adjustment a and mean
#reversion level b
def muOU(x,t,a,b):
    return(a*(b-x))
    
#sigma function for OU process with dispersion sig
def sigmaOU(x,t,sig):
    return(sig)
    
"""Geometric Brownian Motion and Stock Stuff"""
#find MLE estimate of mu + .h sigma^2 and sigma, return confidence intervals
def gbmMLE(data,dt,alpha):
    logs = np.log(data)
    diffLogs = np.diff(logs)
    n = len(diffLogs)
    wMLE = np.mean(diffLogs)/dt
    varMLE = np.var(diffLogs)/dt
    
    #calculate 100 - alpha % CIs
    zAlpha = stat.norm.ppf(1-alpha/2)
    wCI = [wMLE- zAlpha*sqrt(varMLE)/sqrt(n),wMLE \
                + zAlpha*sqrt(varMLE)/sqrt(n)]
    chi2 = stat.chi2(df = n)
    varCI = [n*varMLE / chi2.ppf(1-alpha/2), n*varMLE / chi2.ppf(alpha/2) ]
    return(wMLE,varMLE,wCI,varCI)
    
#simulate geobm with drift m and dispersion sig and the stock it models   
def plotStockSim(num,m,sig,stock,dt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = len(stock)
    ts = np.linspace(0,t-1,t/dt)
    for i in range(0,num):
        ax.plot(ts,geoBrownian(stock[0],m,sig,t,dt),alpha = .4,color = 'k')
    ax.plot(ts,stock,color = 'red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='x',which='both',top='off')
    plt.tick_params(axis='y',which='both',right='off')
    plt.grid(linestyle='-', color = 'black', alpha = .5)
    ax.xaxis.grid(False)
    ax.set_ylabel("Value")
    plt.show()

#test whether the MLE Geometric BM model is a good fit
def checkfit(past,period_length,dt):
    past = past[-(2*period_length):]
    split = np.split(past,2)
    train = split[0]
    test = split[1]
    w,var = gbmMLE(train,dt)
    final_val = np.log(test[-1])
    norm = stat.norm(np.log(test[0])\
            +w*period_length,sqrt(var)*sqrt(period_length))
    if final_val < w * period_length:
        pval = norm.cdf(final_val)
    else:
        pval = 1 - norm.cdf(final_val)
    
    return(pval)
    
#find probability that stock will have positive returns by fitting 
#MLE estimates and modeling stock price as GBM
def probPos(data,period_length,dt):
    
    train = data[-period_length:]
    w,var = gbmMLE(train,dt)
    norm = stat.norm(w*period_length,sqrt(var)*sqrt(period_length))
    prob = 1 - norm.cdf(0)
    return prob