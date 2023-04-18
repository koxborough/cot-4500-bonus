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


if __name__ == "__main__":
    A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
    b = np.array([1, 3, 0])
    tolerance = 1e-6
    iterations = 50

    # Task One: determine number of iterations for Gauss-Seidel to converge
    print(gauss_seidel(A, b, tolerance, iterations))