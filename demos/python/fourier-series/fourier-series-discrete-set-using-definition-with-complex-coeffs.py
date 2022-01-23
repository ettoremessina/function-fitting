import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#t is the independent variable
P = 2. #period value
BT=-6. #initian value of t (begin)
ET=6. #final value of t (end))
FS=1000. #number of discrete values of t between BT and ET

f = lambda t: ((t % P) - (P / 2.)) ** 3 #the periodic real-valued function f(t) with period equal to P to simulate the dataset acquiring
t_range = np.arange(BT, ET, 1/FS) #all discrete values of t in the interval from BT and ET
y_true = f(t_range) #the true f(t)

def compute_complex_fourier_coeffs_from_discrete_set(y_dataset, N): #via Riemann sum; N is up to nthHarmonic
    result = []
    T = len(y_dataset)
    t = np.arange(T)
    for n in range(-N, N+1):
        cn = (1./T) * (y_dataset * np.exp(-1j * 2 * np.pi * n * t / T)).sum()
        result.append(cn)
    return np.array(result)

def fit_func_by_fourier_series_with_complex_coeffs(t, C): #function that computes the Fourier series using an and bn coefficients
    result = 0. + 0.j
    L = int((len(C) - 1) / 2)
    for n in range(-L, L+1):
        c = C[n+L]
        result +=  c * np.exp(1j * 2. * np.pi * n * t / P)
    return result

FDS=20. #number of discrete values of the dataset (that is long as a period)
t_period = np.arange(0, P, 1/FDS)
y_dataset = f(t_period) #generation of discrete dataset

maxN=8
COLs = 2 #cols of plt
ROWs = 1 + (maxN-1) // COLs #rows of plt
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(ROWs, COLs)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle('acquire dataset with period P=' + str(P))

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
for N in range(1, maxN + 1):
    C = compute_complex_fourier_coeffs_from_discrete_set(y_dataset, N) #C contains the list of cn complex coefficients for n in 1..N interval.
    y_approx = fit_func_by_fourier_series_with_complex_coeffs(t_range, C) #y_approx contains the discrete values of approximation obtained by the Fourier series

    row = (N-1) // COLs
    col = (N-1) % COLs
    axs[row, col].set_title('case N=' + str(N))
    axs[row, col].scatter(t_range, y_true, color='blue', s=1, marker='.')
    axs[row, col].scatter(t_range, y_approx, color='red', s=2, marker='.')
plt.show()
