import numpy as np

def norm(x, xo, tolerance):
    return (max(abs(x - xo))) / (max(abs(xo)) + tolerance)

def gauss_seidel(A, b, tolerance, iterations):
    length = len(b)
    x = np.zeros((length), dtype=np.double)
    
    k = 1
    while (k <= iterations):
        xo = x.copy()

        for i in range(length):
            first_sum = second_sum = 0
            
            for j in range(i):
                first_sum += (A[i][j] * x[j])
            
            for j in range(i + 1, length):
                second_sum += (A[i][j] * (xo[j]))

            x[i] = (1 / A[i][i]) * (-first_sum - second_sum + b[i])

            if (norm(x, xo, tolerance) < tolerance):
                return k
            
        k += 1
    
    return k

def jacobi(A, b, tolerance, iterations):
    length = len(b)
    x = np.zeros((length), dtype=np.double)
    
    k = 1
    while (k <= iterations):
        xo = x.copy()

        for i in range(length):
            sum = 0
            for j in range(length):
                if j != i:
                    sum += (A[i][j] * xo[j])

            x[i] = (1 / A[i][i]) * (-sum + b[i])

            if (norm(x, xo, tolerance) < tolerance):
                return k
        
        k += 1
    
    return k

def f(x):
    return ((x ** 3) - (x ** 2) + 2)

def f_prime(x):
    return ((3 * (x ** 2)) - (2 * x))

def newton_raphson(approx, tolerance):
    i = 1
    while (f(approx) != f_prime(approx)):
        p = approx - (f(approx) / f_prime(approx))

        if (abs(p - approx) < tolerance):
            return i
        
        i += 1
        approx = p


if __name__ == "__main__":
    A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
    b = np.array([1, 3, 0])
    tolerance = 1e-6
    iterations = 50

    # Task One: determine number of iterations for Gauss-Seidel to converge
    print(gauss_seidel(A, b, tolerance, iterations))
    print()

    # Task Two: determine number of iterations for Jacobi method to converge
    print(jacobi(A, b, tolerance, iterations))
    print()

    # Task Three
    print(newton_raphson(0.5, tolerance))