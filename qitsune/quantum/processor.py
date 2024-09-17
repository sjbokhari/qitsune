import qiskit
import qiskit.utils
import qiskit_aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

import numpy as np
import cvxpy as cp

# Define the portfolio optimization problem for quantum approach
def create_portfolio_optimization_problem(returns, covariances, budget, num_assets):
    """Create a QuadraticProgram for the portfolio optimization problem."""
    qp = QuadraticProgram()

    # Add variables (0 or 1 for each asset)
    for i in range(num_assets):
        qp.binary_var(f'x{i}')

    # Add the objective function (maximize returns)
    qp.maximize(linear=returns, quadratic=covariances)

    # Add the budget constraint (select exactly 'budget' assets)
    qp.linear_constraint(linear=[1] * num_assets, sense='==', rhs=budget)

    return qp

# Define the classical Markowitz portfolio optimization function
def classical_portfolio_optimization(returns, covariances, budget):
    """Solve the portfolio optimization problem using the classical Markowitz model."""
    num_assets = len(returns)
    x = cp.Variable(num_assets, integer=True)  # Binary variables (0 or 1)
    
    # Define the objective function
    objective = cp.Maximize(returns @ x - cp.quad_form(x, covariances))
    
    # Define the constraints
    constraints = [cp.sum(x) == budget,  # Budget constraint
                   x >= 0, x <= 1]  # Binary constraint
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CPLEX)
    
    return x.value

# Define the returns, covariances, and budget
returns = np.array([0.24, 0.28, 0.30, 0.22, 0.26])  # Example returns for 5 assets
covariances = np.array([[0.1, 0.02, 0.04, 0.05, 0.03],  # Example covariance matrix
                        [0.02, 0.08, 0.02, 0.03, 0.01],
                        [0.04, 0.02, 0.09, 0.04, 0.05],
                        [0.05, 0.03, 0.04, 0.08, 0.02],
                        [0.03, 0.01, 0.05, 0.02, 0.07]])
budget = 2  # Number of assets to select
num_assets = len(returns)

# Quantum Approach
portfolio_problem = create_portfolio_optimization_problem(returns, covariances, budget, num_assets)
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(portfolio_problem)

backend = qiskit_aer.Aer.get_backend('qasm_simulator')
optimizer = COBYLA(maxiter=1000)
sampler = Sampler()
qaoa = QAOA(sampler, optimizer=optimizer, reps=3)
#quantum_instance = QuantumInstance(backend, shots=1024)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

quantum_result = qaoa_optimizer.solve(qubo)

# Classical Approach
classical_result = classical_portfolio_optimization(returns, covariances, budget)

# Output the results
print("Quantum Approach Optimal Portfolio:")
for i in range(num_assets):
    if quantum_result.x[i] > 0:
        print(f"Asset {i + 1} is included in the portfolio.")

quantum_return = sum(quantum_result.x[i] * returns[i] for i in range(num_assets))
print(f"Expected return of the quantum optimized portfolio: {quantum_return}\n")

print("Classical Approach Optimal Portfolio:")
for i in range(num_assets):
    if classical_result[i] > 0:
        print(f"Asset {i + 1} is included in the portfolio.")

classical_return = sum(classical_result[i] * returns[i] for i in range(num_assets))
print(f"Expected return of the classical optimized portfolio: {classical_return}")
