import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.fftpack as spf

#x and y arr the independent variables
PX = 3.*np.pi #period value on x
PY = 3.*np.pi #period value on y
BX=-4.*np.pi #initian value of x,
BY=-4.*np.pi #initian value of y
EX= 4.*np.pi #final value of x
EY= 4.*np.pi #final value of y
FSX=80 #number of discrete values of t between BX and EX
FSY=80 #number of discrete values of t between BY and EY

#the periodic real-valued function f(x,y) with periods PX and PY
f = lambda x, y: (x % PX) - (y % PY)

#all discrete values of x,y in the rectangle from [BX, EX] x [BY, EY]
x_range = np.linspace(BX, EX, FSX)
y_range = np.linspace(BY, EY, FSY)

xyz_true = np.array([[x, y, f(x, y)] for x in x_range for y in y_range])
#xyz_true[:, 2] contains the discrete values of f(x,y) in the rectangle

#function that computes the complex Fourier coefficients c-N,.., c0, .., cN by Discrete Fast Fourier Transform
def compute_complex_fourier_coeffs_from_discrete_set_by_fft2(z_dataset, N): #via tff N is up to nthHarmonic
    z_ds_transf = spf.fft2(z_dataset)

    KX = len(z_dataset)
    KY = len(z_dataset[0])
    K = min(KX, KY)
    if N % 2 == 0:
        if N >= K // 2:
            raise Exception(f"Argument exception: 'N' cannot be >= {K//2}")
    else:
        if N > K // 2:
            raise Exception(f"Argument exception: 'N' cannot be > {K//2}")

    result = []
    for n1 in range(-N, N+1):
        nested = []
        for n2 in range(-N, N+1):
            cn = (1./KX) * (1./KY) * z_ds_transf[n1, n2]
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

FDS=80 #number of discrete values of the dataset (that is long as a period)
x_period = np.arange(0, PX, PX/FDS)
y_period = np.arange(0, PY, PY/FDS)
#generation of discrete dataset
z_dataset = [[f(x, y) for y in y_period] for x in x_period]

plt.rcParams['font.size'] = 8
fig = plt.figure()
fig.suptitle('simulated superficial periodic discrete dataset')

#plot, in the range from BT to ET, the true f(t) in blue and the approximation in red
N = 16
C = compute_complex_fourier_coeffs_from_discrete_set_by_fft2(z_dataset, N)
#C contains the list of cn complex coefficients for n in 1..N interval.

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
