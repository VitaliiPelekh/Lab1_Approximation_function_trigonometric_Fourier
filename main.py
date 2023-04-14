import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import warnings
from scipy.integrate import IntegrationWarning


def f_x(x):
    return x**20 * np.exp(-x**2 / 20)


def a_coefficient(k):
    def integrand(x):
        return f_x(x) * np.cos(k * x)
    return (1 / np.pi) * quad(integrand, -np.pi, np.pi)[0]


def b_coefficient(k):
    def integrand(x):
        return f_x(x) * np.sin(k * x)
    return (1 / np.pi) * quad(integrand, -np.pi, np.pi)[0]


def fourier_series_approximation(x, N):
    sum_a = a_coefficient(0) / 2
    sum_b = 0
    for k in range(1, N + 1):
        sum_a += a_coefficient(k) * np.cos(k * x)
        sum_b += b_coefficient(k) * np.sin(k * x)
    return sum_a + sum_b


def relative_error(x, N):
    return np.abs((f_x(x) - fourier_series_approximation(x, N)) / f_x(x))


def save_results(N):
    a_coeffs = [a_coefficient(k) for k in range(N + 1)]
    b_coeffs = [b_coefficient(k) for k in range(1, N + 1)]

    with open('results.txt', 'w') as f:
        f.write(f"Обчислення наближеня за допомогою ряду Фур'є значення функції x^20 * e^(-x^2 / 20)\n")
        f.write(f'Порядок N = {N}\n\n')
        f.write('Коефіцієнти ряду Фур’є:\n  №\t\t\t\ta_k\t\t\t\tb_k\n')
        for k in range(N + 1):
            a_k_str = f'{a_coeffs[k]:.4f}'
            if k == 0:
                f.write(f'  {k:<4}    {a_k_str:<20}Не існує\n')
            else:
                b_k_str = f'{b_coeffs[k - 1]:.1f}'
                f.write(f'  {k:<4}    {a_k_str:<20}{b_k_str:<12}\n')


def print_results(N):
    print("Обчислення наближеня за допомогою ряду Фур'є значення функції x^20 * e^(-x^2 / 20)\n")
    print(f"Порядок N = {N}\n")
    print("Коефіцієнти ряду Фур'є")
    print("№\t\t\t\ta_k\t\t\t\t\tb_k")

    for k in range(N + 1):
        a_k = a_coefficient(k)
        b_k = b_coefficient(k) if k != 0 else "Не існує"
        print(f"{k:<7} {a_k:<25} {b_k:<12}")

    error = relative_error(np.pi, N)
    print(f"\nПохибка наближення: {error:.4e}")


def plot_harmonics(N):
    x = np.linspace(-np.pi, np.pi, 1000)
    for k in range(N + 1):
        plt.plot(x, a_coefficient(k) * np.cos(k * x) + b_coefficient(k) * np.sin(k * x), label=f'k = {k}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Harmonic Functions')
    plt.title('Fourier Harmonics')
    plt.show()


def plot_coefficients(N):
    a_coeffs = [a_coefficient(k) for k in range(N + 1)]
    b_coeffs = [b_coefficient(k) for k in range(1, N + 1)]

    plt.plot(range(N + 1), a_coeffs, marker='o', label='a(k)')
    plt.plot(range(1, N + 1), b_coeffs, marker='x', label='b(k)')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('Coefficients')
    plt.title('Fourier Coefficients in Frequency Domain')
    plt.show()


def plot_sequential_approximation(x, N):
    plt.figure()
    plt.plot(x, f_x(x), label='f(x)', linewidth=2)

    for i in [int(N * 0.25), int(N * 0.5), N]:
        plt.plot(x, fourier_series_approximation(x, i), label=f'N = {i}', linestyle='--')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Sequential Fourier Series Approximations')
    plt.show()


def plot_relative_error(x, N):
    errors = [relative_error(val, N) for val in x]

    plt.plot(x, errors)
    plt.xlabel('x')
    plt.ylabel('Relative Error')
    plt.title('Relative Error of Fourier Series Approximation')
    plt.show()


def plot_function_and_approximation(x, N):
    plt.figure()
    plt.plot(x, f_x(x), label='f(x)', linewidth=2)
    plt.plot(x, fourier_series_approximation(x, N), label=f'Approximation N = {N}', linestyle='--')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function f(x) and Fourier Series Approximation')
    plt.show()


def main(N):
    x = np.linspace(-np.pi, np.pi, 1000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)

        plot_function_and_approximation(x, N)
        plot_sequential_approximation(x, N)
        plot_coefficients(N)
        plot_harmonics(N)
        plot_relative_error(x, N)
        print_results(N)
        save_results(N)


if __name__ == '__main__':
    N = 10  # Порядок наближення
    main(N)
