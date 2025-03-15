import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import math

# x and y data points
x_data = np.array([1972, 1158, 1006, 941, 900, 873, 854])
y_data = np.array([0.000000110982470, 0.000003337961974, 0.000013354700855, 0.000039086929331, 0.000101214574899, 0.000180375180375, 0.000320512820513])

def power_law(x, a, b):
    """Power law function: y = a * x^(-b)"""
    return a * x**(-b)

def hyperbolic(x, a, b):
    """Hyperbolic function: y = a / (b + x)"""
    return a / (b + x)

def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def rational_function(x, a, b, c, d):
    """Rational function: y = a / (b + c*x + d*x²)"""
    return a / (b + c*x + d*x**2)

def inverse_quadratic(x, a, b, c):
    """Inverse quadratic: y = a / (b + c*x²)"""
    return a / (b + c*x**2)

# List of models to try (with initial parameter guesses)
models = [
    {'name': 'Power Law', 'function': power_law, 'p0': [1e5, 3]},
    {'name': 'Hyperbolic', 'function': hyperbolic, 'p0': [0.1, -700]},
    {'name': 'Exponential Decay', 'function': exponential_decay, 'p0': [0.01, 0.005, 0]},
    {'name': 'Rational Function', 'function': rational_function, 'p0': [10, -700, 0.1, 0.0001]},
    {'name': 'Inverse Quadratic', 'function': inverse_quadratic, 'p0': [100, -7e5, 1]},
]

# Function to evaluate and plot model fits
def fit_and_evaluate_model(model, x_data, y_data, extended_x=None):
    try:
        # Fit the model
        params, cov = curve_fit(model['function'], x_data, y_data, p0=model['p0'], maxfev=10000)
        
        # Generate prediction points for smooth curve
        if extended_x is None:
            extended_x = np.linspace(min(x_data)*0.9, max(x_data)*1.1, 1000)
        y_pred = model['function'](extended_x, *params)
        
        # Calculate fit metrics
        y_fit = model['function'](x_data, *params)
        r2 = r2_score(y_data, y_fit)
        rmse = math.sqrt(mean_squared_error(y_data, y_fit))
        
        return {
            'name': model['name'],
            'params': params,
            'cov': cov,
            'x_extended': extended_x,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse,
            'function': model['function']
        }
    except Exception as e:
        print(f"Failed to fit {model['name']}: {e}")
        return None

# Generate extended x-values for plotting
extended_x = np.linspace(800, 2500, 1000)

# Fit all models
results = []
for model in models:
    result = fit_and_evaluate_model(model, x_data, y_data, extended_x)
    if result:
        results.append(result)

# Sort results by goodness of fit (R²)
results.sort(key=lambda x: x['r2'], reverse=True)

plt.figure(figsize=(12, 10))

# Plot the original data points
plt.scatter(x_data, y_data, color='blue', label='Original Data', zorder=5, s=80)

# Plot the top 3 best fitting models
colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
for i, result in enumerate(results[:3]):
    plt.plot(result['x_extended'], result['y_pred'], 
             label=f"{result['name']}", 
             color=colors[i], linewidth=2)

# Set log scale for y-axis for better visualization
plt.yscale('log')

# Add labels and title
plt.xlabel('Distance from camera y-axis (pixels)', fontsize=12)
plt.ylabel('Scaling Factor (m²/pixel²)', fontsize=12)
plt.title('Curve Fitting for Scaling Factor Value Based on y-Distance from Camera', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=10)

# Create a results table
results_data = {
    'Model': [r['name'] for r in results],
    'R²': [r['r2'] for r in results],
    'RMSE': [r['rmse'] for r in results]
}
results_df = pd.DataFrame(results_data)

# Display results
print("Model Fitting Results (sorted by goodness of fit):")
print(results_df)
print("\nBest Model Parameter Values:")
best_model = results[0]
params = best_model['params']
param_names = best_model['function'].__code__.co_varnames[1:len(params)+1]
for name, value in zip(param_names, params):
    print(f"{name} = {value}")

print("\nFunction Formula:")
if best_model['name'] == 'Power Law':
    print(f"y = {params[0]:.6e} * x^(-{params[1]:.6f})")
elif best_model['name'] == 'Hyperbolic':
    print(f"y = {params[0]:.6e} / ({params[1]:.6f} + x)")
elif best_model['name'] == 'Exponential Decay':
    print(f"y = {params[0]:.6e} * exp(-{params[1]:.6f} * x) + {params[2]:.6e}")
elif best_model['name'] == 'Rational Function':
    print(f"y = {params[0]:.6e} / ({params[1]:.6f} + {params[2]:.6e}*x + {params[3]:.6e}*x²)")
elif best_model['name'] == 'Inverse Quadratic':
    print(f"y = {params[0]:.6e} / ({params[1]:.6f} + {params[2]:.6e}*x²)")
elif best_model['name'] == 'Piecewise Power-Rational':
    print(f"y = {params[0]:.6e} * x^(-{params[1]:.6f}) for x < {params[5]:.2f}")
    print(f"y = {params[2]:.6e} / ({params[3]:.6f} + {params[4]:.6e}*x) for x >= {params[5]:.2f}")

plt.tight_layout()
plt.savefig('scaling_factor_curve_fit.png', dpi=300)
plt.show()