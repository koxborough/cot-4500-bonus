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

def hermite_matrix(x_points, y_points, derivative):
    size = 2 * len(x_points)
    matrix = np.zeros((size, size + 1))

    for i in range(size):
        matrix[i][0] = x_points[i // 2]
        matrix[i][1] = y_points[i // 2]

    for i in range(1, size, 2):
        matrix[i][2] = derivative[i // 2]

    for i in range(2, size):
        for j in range(2, i + 2):
            if matrix[i][j] != 0.:
                continue
            
            matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (matrix[i][0] - matrix[i - j + 1][0])

    matrix = np.delete(matrix, size, 1)
    
    return matrix


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

    # Task Three: determine number of iterations for Newton-Raphson
    print(newton_raphson(0.5, tolerance))
    print()

    # Task Four: find Hermite polynomial approximation matrix
    print(np.matrix(hermite_matrix([0., 1., 2.], [1., 2., 4.], [1.06, 1.23, 1.55])))