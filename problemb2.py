import numpy as np
import matplotlib.pyplot as plt
import math as m

def Gaussian(sig,T):
    (a,b)=(int(-5/T),int(5/T))
    n = list(range(a,b))
    X = [0]*len(n)
    for i in range(0,len(n)):
        X[i]= m.exp(-1*(((i-((5/T)))*T)**2)/(2*sig))
    return X

def DTFT(X,Npoint):
    N = len(X)
    w=np.linspace(-m.pi,m.pi,Npoint)
    n=list(range(0,N))
    dtft= [0]*Npoint
    
    for i in range(0,Npoint-1):
        w_i = w[i]
        a=[]
        for k in n:
            b= -1j*w_i*k
            a.append(b)
        exp_arr= np.exp(a)
        Product=[]
        for t in range(0,Npoint-1):
            b = X[t]*exp_arr[t]
            Product.append(b)
        dtft[i]= np.sum(Product)
    return(dtft)

k=0
for sig in[0.1,1]:
    for T in [0.001,0.01,0.1,1]:

        g_1 = Gaussian(sig,T)
        N = list(range(int(-5/T),int(5/T)))
        n_point = len(g_1)
        gaussian_dtft = DTFT(g_1,n_point)
        W = np.linspace(-m.pi,m.pi,n_point)
        
        fig,axis = plt.subplots(2,num=k+1)
        absolute =[abs(elem) for elem in gaussian_dtft]
        axis[0].plot(W,absolute)
        axis[0].set_title(f"magnitude response of Gaussian with T = {T} and sig = {sig}")
        axis[0].set_xlabel("Frequency in Radian")
        axis[0].set_ylabel("magnitude")
        axis[1].plot(W,np.angle(gaussian_dtft))
        axis[1].set_title(f"phase response Gaussian with T = {T} and sig = {sig}")
        axis[1].set_xlabel("Frequency in Radian")
        axis[1].set_ylabel("phase angle")

        k+=1

def exp_abs(T):
    n = list(range(int(-5/T),int(5/T)))
    X = [0]*len(n)
    for i in range(0,len(n)):
        X[i]= m.exp(-1*abs(n[i]*T))
    return X

k=0

for T in [0.001,0.01,0.1,1]:
    e_1 = exp_abs(T)
    N = list(range(int(-5/T),int(5/T)))
    n_point = len(e_1)
    exp_abso_dtft = DTFT(e_1,n_point)
    W = np.linspace(-m.pi,m.pi,n_point)
        
    fig,axis = plt.subplots(2,num=k+1)
    absolute =[abs(elem) for elem in exp_abso_dtft]
    axis[0].plot(W,absolute)
    axis[0].set_title(f"magnitude response of Gaussian with T = {T} ")
    axis[0].set_xlabel("Frequency in Radian")
    axis[0].set_ylabel("magnitude")
    axis[1].plot(W,np.angle(exp_abso_dtft))
    axis[1].set_title(f"phase response Gaussian with T = {T} ")
    axis[1].set_xlabel("Frequency in Radian")
    axis[1].set_ylabel("phase angle")

    k+=1

def unit_step(n):
    if n>=0:
        return(1)
    else:
        return(0)

def exp_cos_unit(T):
    n = list(range(int(-5/T),int(5/T)))
    X = [0]*len(n)
    for i in range(0,len(n)):
        X[i] = m.exp(-0.02*(i-(5/T))*T)*m.cos(200*m.pi*(i-(5/T))*T)*unit_step(n[i])
    return(X)   

k=0

for T in [0.001,0.01,0.1,1]:
    ec_1 = exp_cos_unit(T)
    N = list(range(int(-5/T),int(5/T)))
    n_point = len(ec_1)
    exp_cos_unit_dtft = DTFT(ec_1,n_point)
    W = np.linspace(-m.pi,m.pi,n_point)
        
    fig,axis = plt.subplots(2,num=k+1)
    absolute =[abs(elem) for elem in exp_cos_unit_dtft]
    axis[0].plot(W,absolute)
    axis[0].set_title(f"magnitude response of Gaussian with T = {T} ")
    axis[0].set_xlabel("Frequency in Radian")
    axis[0].set_ylabel("magnitude")
    axis[1].plot(W,np.angle(exp_cos_unit_dtft))
    axis[1].set_title(f"phase response Gaussian with T = {T} ")
    axis[1].set_xlabel("Frequency in Radian")
    axis[1].set_ylabel("phase angle")

    k+=1         
