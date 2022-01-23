import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#t is the independent variable
P = 2. #period value
BT=-6. #initian value of t (begin)
ET=6. #final value of t (end))
FS=1000. #number of discrete values of t between BT and ET

f = lambda t: ((t % P) -P/2) ** 2 + (1j * ((t % P) -P/2)) #the periodic complex-valued function f(t) with period equal to P
t_range = np.arange(BT, ET, 1/FS) #all discrete values of t in the interval from BT and ET
y_true = f(t_range) #the true f(t)
y_true_real = [y.real for y in y_true]
y_true_imag = [y.imag for y in y_true]

def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = spi.quad(real_func, a, b, **kwargs)
    imag_integral = spi.quad(imag_func, a, b, **kwargs)
    integral = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return integral

def compute_complex_fourier_coeffs(func, N): #function that computes the real fourier couples of coefficients (a1, b1)...(aN, bN)
    result = []
    for n in range(-N, N+1):
        cn = (1./P) * complex_quad(lambda t: func(t) * np.exp(-1j * 2 * np.pi * n * t / P), 0, P)[0]
        result.append(cn)
    return np.array(result)

def fit_func_by_fourier_series_with_complex_coeffs(t, C): #function that computes the Fourier series using an and bn coefficients
    result = 0. + 0.j
    L = int((len(C) - 1) / 2)
    for n in range(-L, L+1):
        c = C[n+L]
        result +=  c * np.exp(1j * 2. * np.pi * n * t / P)
    return result

maxN=8
COLs = 2 #cols of plt
ROWs = maxN  #rows of plt
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(ROWs, COLs)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle('f(t) = ((t % P) -P/2) ** 2 + (1j * ((t % P) -P/2)) where P=' + str(P))

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
for N in range(1, maxN + 1):
    C = compute_complex_fourier_coeffs(f, N) #C contains the list of cn complex coefficients for n in 1..N interval.
    y_approx = fit_func_by_fourier_series_with_complex_coeffs(t_range, C) #y_approx contains the discrete values of approximation obtained by the Fourier series
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
