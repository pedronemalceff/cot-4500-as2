import numpy as np 

#Question 1
def neville_method(x, data):
    # data is a list of (x, f(x)) pairs
    n = len(data)
    f = [d[1] for d in data]
    for i in range(1, n):
        for j in range(n-i):
            f[j] = ((x-data[j+i][0])*f[j] + (data[j][0]-x)*f[j+1])/(data[j][0]-data[j+i][0])
    return f[0]

data = [(3.6, 1.675), (3.8, 1.436), (3.9, 1.318)]
x = 3.7
result = neville_method(x, data)

print()
print(result)

#Question 2 
# Define the data
x = [7.2, 7.4, 7.5, 7.6]
y = [23.5492, 25.3913, 26.8224, 27.4589]

# Construct the forward difference table
n = len(x)
d = [[0 for j in range(n)] for i in range(n)]
d[0] = y

for i in range(1, n):
    for j in range(n-i):
        d[i][j] = (d[i-1][j+1] - d[i-1][j]) / (x[i+j] - x[j])

# Construct the polynomial approximations
p3 = [d[1][0], d[2][0], d[3][0]]

# Print the polynomial coefficients
print()
print(p3)  # [9.210500000000001, 17.00166666666675, -141.82916666666722]

#Question 3

# Define the value at which to approximate the function
x0 = 7.3

# Compute the value of the polynomial at x0
f_approx = y[0] + p3[0]*(x0-x[0]) + p3[1]*(x0-x[0])*(x0-x[1]) + p3[2]*(x0-x[0])*(x0-x[1])*(x0-x[2]) 

# Print the result
print()
print(f_approx) # 24.016574999999992

#Question 4

# Given data points
x = np.array([3.6, 3.8, 3.9])
y = np.array([1.675, 1.436, 1.318])
y_prime = np.array([-1.195, -1.188, -1.182])

# Compute divided differences
f = np.zeros((2 * len(x), 2 * len(x)))
f[:, 0] = np.repeat(x, 2)
f[:, 1] = np.repeat(y, 2)

# Populate the divided difference table
for j in range(2, 2 * len(x)):
    for i in range(j - 1, 2 * len(x) - j + 1):
        if f[i, 0] == f[i - j + 1, 0]:
            f[i, j] = y_prime[i // 2]
        else:
            f[i, j] = (f[i, j - 1] - f[i - 1, j - 1]) / (f[i, 0] - f[i - j + 1, 0])

#Function broke down a little so I had to manually calculate these values with the formula
f[5,2] = y_prime[2]
f[4,3] = (f[4,2] - f[3,2]) / (f[4,0] - f[3,0])
f[5,3] = (f[5,2] - f[4,2]) / (f[5,0] - f[3,0])
f[3,4] = (f[3,3] - f[2,3]) / (f[3,0] - f[1,0])
f[4,4] = (f[4,3] - f[3,3]) / (f[4,0] - f[1,0])
f[5,4] = (f[5,3] - f[4,3]) / (f[4,0] - f[2,0])
f[4,5] = (f[4,4] - f[3,4]) / (f[4,0] - f[1,0])
f[5,5] = (f[5,4] - f[4,4]) / (f[4,0] - f[1,0])


# # Print the Hermite polynomial approximation matrix
np.set_printoptions(precision=7, suppress=True, linewidth=100)
print()
print(f)

#Question 5

# input data
x = np.array([2, 5, 8, 10])
y = np.array([3, 5, 7, 9])

#Matrix A
#Calculations
h0 = x[1]-x[0]
h1 = x[2]-x[1]
h2 = x[3]-x[2]
a0 = y[0]
a1 = y[1]
a2 = y[2]
a3 = y[3]

#Initialzing array 
matrix_a = np.zeros((4, 4))
matrix_a[0,0] = 1
matrix_a[3,3] = 1

#Populating arrray
matrix_a[1,0] = h0
matrix_a[1,1] = 2 * (h0+h1)
matrix_a[1,2] = h1

matrix_a[2,1] = h1
matrix_a[2,2] = 2 * (h1+h2)
matrix_a[2,3] = h2

#Print matrix
print()
print(matrix_a)

#Vector b

vector_b = np.zeros((1, 4))

vector_b[0,1] = (3/h1) * (a2-a1) - (3/h0) * (a1-a0)
vector_b[0,2] = (3/h2) * (a3-a2) - (3/h1) * (a2-a1)

#Print matrix
print()
print(vector_b)

#Vector x

# Use linalg.solve() to perform matrix division
C = np.linalg.solve(matrix_a, vector_b.T)

# Print the resulting matrix C
print()
print(C.flatten())




