import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#x and y arr the independent variables
PX= 3.#period value on x
PY= 3. #period value on y
BX=-6. #initian value of x,
BY=-6. #initian value of y
EX= 6. #final value of x
EY= 6. #final value of y
FSX=80 #number of discrete values of t between BX and EX
FSY=80 #number of discrete values of t between BY and EY

#the periodic real-valued function f(x,y) with periods PX and PY
f = lambda x, y: (x % PX) - (y % PY)

#all discrete values of x,y in the rectangle from [BX, EX] x [BY, EY]
x_range = np.linspace(BX, EX, FSX)
y_range = np.linspace(BY, EY, FSY)

xyz_true = np.array([[x, y, f(x, y)] for x in x_range for y in y_range])
#xyz_true[:, 2] contains the discrete values of f(x,y) in the rectangle

#function to integrate on complex field
def complex_dblquad(func, a, b, gfun, hfun, **kwargs):
    def real_func(x, y):
        return np.real(func(x, y))
    def imag_func(x, y):
        return np.imag(func(x, y))
    real_integral = spi.dblquad(real_func, a, b, gfun, hfun, **kwargs)
    imag_integral = spi.dblquad(imag_func, a, b, gfun, hfun, **kwargs)
    integral = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return integral

#function that computes the complex fourier coefficients c-N,.., c0, .., cN
def compute_complex_fourier_coeffs(func, N):
    result = []
    for n1 in range(-N, N+1):
        nested = []
        for n2 in range(-N, N+1):
            cn = (1./PX) * (1./PY) * complex_dblquad(
                lambda y, x: func(x, y) \
                    * np.exp(-1j * 2 * np.pi * n1 * x / PX) \
                    * np.exp(-1j * 2 * np.pi * n2 * y / PX),
                0, PX, lambda x: 0, lambda x: PY)[0]
            nested.append(cn)
        result.append(np.array(nested))
    return np.array(result)

#function that computes the complex form Fourier series using cn coefficients in 2 variables
def fit_func2var_by_fourier_series_with_complex_coeffs(x, y, C):
    result = 0. + 0.j
    L = int((len(C) - 1) / 2)
    for n1 in range(-L, L+1):
        for n2 in range(-L, L+1):
            c = C[n1+L][n2+L]
            result +=  c \
                * np.exp(1j * 2. * np.pi * n1 * x / PX) \
                * np.exp(1j * 2. * np.pi * n2 * y / PY)
    return result

N=16
plt.rcParams['font.size'] = 8
fig = plt.figure()
fig.suptitle('f(x, y) = (x mod PX) - (y mod PY) where PX=3 and PY=3')

C = compute_complex_fourier_coeffs(f, N)
#C contains the matrix of cn coefficients for (n1, n2) in [1, N] x [1, N]???

xyz_approx = []
for x in x_range:
    for y in y_range:
        xyz_approx.append([x, y, fit_func2var_by_fourier_series_with_complex_coeffs(x, y, C)])
xyz_approx = np.array(xyz_approx)
#xyz_approx[:, 2] contains the discrete values of approximation obtained by the Fourier series

#plot the true f(x,y) in blue and the approximation in red
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter3D(xyz_true[:, 0], xyz_true[:, 1], xyz_true[:, 2], color='blue', s=1, marker='.');
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(xyz_approx[:, 0], xyz_approx[:, 1], xyz_approx[:, 2], color='red', s=1, marker='.');

plt.show()
