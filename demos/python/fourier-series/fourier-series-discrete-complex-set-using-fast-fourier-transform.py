import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.fftpack as spf

#t is the independent variable
P = 3. #period value
BT=-6. #initian value of t (time begin)
ET=6. #final value of t (time end)
FS=1000. #number of discrete values of t between BT and ET

#the periodic complex-valued function f(t) with period equal to P
f = lambda t: ((t % P) - P/2.) ** 2 + ((t % P) -P/2.) * 1j
t_range = np.arange(BT, ET, 1/FS) #all discrete values of t in the interval from BT and ET
y_true = f(t_range) #the true f(t)
y_true_real = [y.real for y in y_true]
y_true_imag = [y.imag for y in y_true]

#function that computes the complex Fourier coefficients c-N,.., c0, .., cN by Discrete Fast Fourier Transform
def compute_complex_fourier_coeffs_from_discrete_set_by_fft(y_dataset, N): #via tff N is up to nthHarmonic
    result = []
    y_ds_transf = spf.fft(y_dataset)

    K = len(y_dataset)
    if N % 2 == 0:
        if N >= K // 2:
            raise Exception(f"Argument exception: 'N' cannot be >= {K//2}")
    else:
        if N > K // 2:
            raise Exception(f"Argument exception: 'N' cannot be > {K//2}")

    for n in range(-N, N+1):
        cn = (1./K) * y_ds_transf[n]
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

maxN=4
COLs = 2 #cols of plt
ROWs = maxN #rows of plt
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(ROWs, COLs)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle('simulated complex dataset with period P=' + str(P))

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
for N in range(1, maxN + 1):
    C = compute_complex_fourier_coeffs_from_discrete_set_by_fft(y_dataset, N)
    #C contains the list of cn complex coefficients for n in 1..N interval.

    y_approx = fit_func_by_fourier_series_with_complex_coeffs(t_range, C)
    #y_approx contains the discrete values of approximation obtained by the Fourier series

    y_approx_real = [y.real for y in y_approx]
    y_approx_imag = [y.imag for y in y_approx]
    row = (N-1)
    axs[row, 0].set_title('real part, case N=' + str(N))
    axs[row, 1].set_title('imag part, case N=' + str(N))
    axs[row, 0].scatter(t_range, y_true_real, color='blue', s=1, marker='.')
    axs[row, 0].scatter(t_range, y_approx_real, color='red', s=2, marker='.')
    axs[row, 1].scatter(t_range, y_true_imag, color='blue', s=1, marker='.')
    axs[row, 1].scatter(t_range, y_approx_imag, color='red', s=2, marker='.')
plt.show()
