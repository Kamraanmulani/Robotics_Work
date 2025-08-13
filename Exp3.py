import sympy as sp

# Define symbolic variables
theta, d, a, alpha = sp.symbols('theta_{i+1} d_{i+1} a_{i+1} alpha_{i+1}')

# Define the Denavit-Hartenberg transformation matrix
DH_matrix = sp.Matrix([
    [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
    [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
    [0, sp.sin(alpha), sp.cos(alpha), d],
    [0, 0, 0, 1]
])

# Display the matrix
sp.pprint(DH_matrix, use_unicode=True)
