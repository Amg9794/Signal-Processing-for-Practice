import numpy as np
import matplotlib.pyplot as plt

# Define the time range
T_values = [0.001, 0.01, 0.1, 1]
t = np.linspace(-5, 5, 1000)

# Define the function
def x1(t,sigma2,T):
    return np.exp((-t**2)/2*sigma2)

# Plot the results
for i, T in enumerate(T_values):
    nT = np.arange(-5, 5, T)
    x_sampled = x1(nT,1, T)   # sigma2 =1
    plt.subplot(2,2,i+1)
    plt.stem(nT, x_sampled, label='T = {}'.format(T))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('T = {}'.format(T))
    plt.legend()
plt.suptitle('Sampled version of exp((-t**2)/2*sigma2))')
plt.show()   


# Define the function
def x2(t, T):
    return np.exp(-abs(t))


# Plot the results
for i, T in enumerate(T_values):
    nT = np.arange(-5, 5, T)
    x_sampled = x2(nT, T)
    plt.subplot(2,2,i+1)
    plt.stem(nT, x_sampled, label='T = {}'.format(T))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('T = {}'.format(T))
    plt.legend()
plt.suptitle('Sampled version of exp(-abs(t))')
plt.show()   

# Define the function
def x3(t, T):
    return np.exp(-0.02*t)*np.cos(2*np.pi*100*t)*(t >= 0)


# Plot the results
for i, T in enumerate(T_values):
    nT = np.arange(-0, 5, T)
    x_sampled = x3(nT, T)
    plt.subplot(2,2,i+1)
    plt.stem(nT, x_sampled, label='T = {}'.format(T))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('T = {}'.format(T))
    plt.legend()
plt.suptitle('Sampled version of exponentially decaying fun')
plt.show()   


