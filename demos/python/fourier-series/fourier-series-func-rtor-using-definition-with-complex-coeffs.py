import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#t is the independent variable
P = 3. #period value
BT=-6. #initian value of t (begin time)
ET=6. #final value of t (end time)
FS=1000. #number of discrete values of t between BT and ET

#the periodic real-valued function f(t) with period equal to P
f = lambda t: ((t % P) - (P / 2.)) ** 3

#all discrete values of t in the interval from BT and ET
t_range = np.arange(BT, ET, 1/FS)
y_true = f(t_range) #the true f(t)

#function to integrate on complex field
def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = spi.quad(real_func, a, b, **kwargs)
    imag_integral = spi.quad(imag_func, a, b, **kwargs)
    integral = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return integral

#function that computes the complex fourier coefficients c-N,.., c0, .., cN
def compute_complex_fourier_coeffs(func, N):
    result = []
    for n in range(-N, N+1):
        cn = (1./P) * complex_quad(lambda t: func(t) * np.exp(-1j * 2 * np.pi * n * t / P), 0, P)[0]
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

maxN=8
COLs = 2 #cols of plt
ROWs = 1 + (maxN-1) // COLs #rows of plt
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(ROWs, COLs)
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
fig.suptitle('f(t) = ((t % P) - (P / 2.)) ** 3 where P=' + str(P))

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
for N in range(1, maxN + 1):
    C = compute_complex_fourier_coeffs(f, N)
    #C contains the list of cn complex coefficients for n in 1..N interval.

    y_approx = fit_func_by_fourier_series_with_complex_coeffs(t_range, C)
    #y_approx contains the discrete values of approximation obtained by the Fourier series

    row = (N-1) // COLs
    col = (N-1) % COLs
    axs[row, col].set_title('case N=' + str(N))
    axs[row, col].scatter(t_range, y_true, color='blue', s=1, marker='.')
    axs[row, col].scatter(t_range, y_approx, color='red', s=2, marker='.')
plt.show()
