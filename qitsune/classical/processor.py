import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns
import cvxpy as cp
import random
from deap import base, creator, tools, algorithms

# 1. Load historical data from Yahoo Finance
tickers = ['AAPL', 'MSFT', 'GOOGL', 'GLMD', 'AMZN', 'NFLX', 'TSLA', 'NVDA', 'BABA', 'JPM']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

# 2. Calculate expected returns and covariance matrix
returns = data.pct_change().dropna()

# Update tickers to match the remaining assets after dropna()
tickers = returns.columns
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Symmetrize the covariance matrix to ensure it's valid
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# --- Optimizations using pypfopt ---

# 3. Define the portfolio optimization problem

# Number of assets
num_assets = len(tickers)

# Portfolio statistics functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    # Calculate portfolio return and variance
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# 4. Sharpe Ratio Maximization (Scipy)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility  # Negative Sharpe Ratio

# Constraint: sum of weights = 1
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

# Bounds: No short selling (weights >= 0)
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess
initial_guess = num_assets * [1. / num_assets]

# Optimization
optimized_sharpe = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix), 
                            method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimized Weights for Maximum Sharpe Ratio:", optimized_sharpe.x)

# 5. Minimum Variance Portfolio (Efficient Frontier)
S = risk_models.sample_cov(returns)
mu = expected_returns.mean_historical_return(data)
ef = EfficientFrontier(mu, S)

weights_min_var = ef.min_volatility()
print("Minimum Variance Weights:", weights_min_var)

# 6. Genetic Algorithm Optimization
# Set up genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_portfolio(individual):
    weights = np.array(individual) / sum(individual)  # Normalize weights
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - 0.01) / p_volatility
    return sharpe_ratio,

toolbox.register("evaluate", eval_portfolio)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic algorithm execution
population = toolbox.population(n=300)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
best_weights = np.array(best_individual) / sum(best_individual)
print("Genetic Algorithm Optimized Weights:", best_weights)

# 7. Visualization of the Efficient Frontier
def plot_efficient_frontier(ef_weights):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate the efficient frontier curve
    ef_returns, ef_volatility = [], []
    for risk_level in np.linspace(0, 1, 100):
        #ef.min_volatility()
        ef_weights = ef.clean_weights()
        ret, vol = portfolio_performance(list(ef_weights.values()), mu, S)
        ef_returns.append(ret)
        ef_volatility.append(vol)

    # Plot
    ax.plot(ef_volatility, ef_returns, label='Efficient Frontier', color='b')
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    plt.show()

plot_efficient_frontier(ef.clean_weights())