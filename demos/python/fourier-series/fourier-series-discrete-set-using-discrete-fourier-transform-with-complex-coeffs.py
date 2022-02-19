import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#t is the independent variable
P = 3. #period value
BT=-6. #initian value of t (time begin)
ET=6. #final value of t (time end)
FS=1000 #number of discrete values of t between BT and ET

#the periodic real-valued function f(t) with period equal to P to simulate an acquired dataset
f = lambda t: ((t % P) - (P / 2.)) ** 3
t_range = np.linspace(BT, ET, FS) #all discrete values of t in the interval from BT and ET
y_true = f(t_range) #the true f(t)

#function that computes the discrete fourier transform
def compute_discrete_fourier_transform(y_dataset, t):
    K = len(y_dataset)
    result = 0. + 0.j
    for k in range(0, K):
        result += y_dataset[k] * np.exp((-1.j * 2 * np.pi * t * k) / K)
    return result

#function that computes the complex Fourier coefficients c-N,.., c0, .., cN by Discrete Fourier Transform
def compute_complex_fourier_coeffs_from_discrete_set_by_dft(y_dataset, N):
    result = []
    K = len(y_dataset)
    for n in range(-N, N+1):
        cn = (1./K) * compute_discrete_fourier_transform(y_dataset, n)
        result.append(cn)
    return np.array(result)

#function that computes the complex form Fourier series using cn coefficients
def fit_func_by_fourier_series_with_complex_coeffs(t, C):
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
fig.suptitle('simulated dataset with period P=' + str(P))

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
for N in range(1, maxN + 1):
    C = compute_complex_fourier_coeffs_from_discrete_set_by_dft(y_dataset, N)
    #C contains the list of cn complex coefficients for n in 1..N interval
    y_approx = fit_func_by_fourier_series_with_complex_coeffs(t_range, C)
    #y_approx contains the discrete values of approximation obtained by the Fourier series

    row = (N-1) // COLs
    col = (N-1) % COLs
    axs[row, col].set_title('case N=' + str(N))
    axs[row, col].scatter(t_range, y_true, color='blue', s=0.15, marker='o')
    axs[row, col].scatter(t_range, y_approx, color='red', s=0.25, marker='o')
plt.show()
