import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == '__main__':
    # Define the cost function and its gradient
    f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    fd = lambda x: np.array([0, 0])  # Placeholder for the gradient, to be implemented

    # Configure optimization parameters
    x_init = [0, 0]  # Try different initial points here
    learn_rate = 0.01  # Try different learning rates (e.g., 0.005, 0.0001)
    max_iter = 1000  # Try different maximum iteration values (e.g., 100, 100000)
    min_tol = 1e-6  # Adjust the tolerance (e.g., 1e-5, 1e-7)

    # Gradient Descent Optimization
    x = np.array(x_init)
    gd_xs = [x]
    for i in range(max_iter):
        # Run the gradient descent
        xp = x
        # Implement your gradient descent step here
        x = x - learn_rate * fd(x)

        gd_xs.append(x)

        # Check the termination condition
        if np.linalg.norm(x - xp) < min_tol:
            break
    gd_xs = np.array(gd_xs)

    # Optimize the cost function using SciPy for comparison
    result = minimize(f, x_init, tol=min_tol, options={'maxiter': max_iter, 'return_all': True})
    sp_xs = np.array(result.allvecs)

    # Visualize the results
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = f((xx, yy))
    plt.contourf(xx, yy, zz)
    plt.plot(gd_xs[:, 0], gd_xs[:, 1], 'r.', label=f'GD (iter={len(gd_xs)})')
    plt.plot(sp_xs[:, 0], sp_xs[:, 1], 'b.', label=f'SciPy (iter={result.nit})')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.legend()
    plt.show()
