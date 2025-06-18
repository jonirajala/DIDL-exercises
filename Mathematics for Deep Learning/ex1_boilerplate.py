# Boilerplate Exercise Code for Mathematics for Deep Learning (D2L.ai Appendix)

# 1. Geometry & Linear Algebra
# Task: Compute vector norms, dot products, projections, and angles
import numpy as np

def vector_operations():
    """
    Returns:
        tuple: (dot_product, norm_u, norm_v, projection of u on v, angle between u and v in radians)
    """
    u = np.array([1.0, 2.0])  # Replace with any 2D vector
    v = np.array([3.0, 4.0])  # Replace with any 2D vector
    # TODO: Compute dot product, norm, projection of u on v, angle between u and v
    raise NotImplementedError 


# 2. Eigendecomposition
# Task: Compute eigenvalues and eigenvectors of a symmetric matrix

def eigen_decomposition():
    """
    Returns:
        tuple: (eigenvalues, eigenvectors) of the matrix A
    """
    A = np.array([[2, -1], [-1, 2]])  # Example symmetric matrix
    # TODO: Compute eigendecomposition of A
    raise NotImplementedError


# 3. Single-variable Calculus
# Task: Numerically compute derivative of a function

def numerical_derivative(f, x, h=1e-5):
    """
    Args:
        f (callable): function of a single variable
        x (float): point at which to evaluate derivative
        h (float): step size for central difference

    Returns:
        float: estimated derivative at x
    """
    # TODO: Implement central difference method
    raise NotImplementedError


# 4. Multivariable Calculus
# Task: Estimate gradient and Hessian of a function at a point

def multivariable_function(x, y):
    # Example: return x**2 * y + np.sin(x * y)
    raise NotImplementedError

def gradient_estimation(f, point, h=1e-5):
    """
    Args:
        f (callable): function of two variables taking list [x, y]
        point (list): [x, y] at which to compute gradient
        h (float): step size

    Returns:
        np.ndarray: estimated gradient vector
    """
    # TODO: Numerically estimate the gradient
    raise NotImplementedError


def hessian_estimation(f, point, h=1e-5):
    """
    Args:
        f (callable): function of two variables taking list [x, y]
        point (list): [x, y] at which to compute Hessian
        h (float): step size

    Returns:
        np.ndarray: 2x2 Hessian matrix
    """
    # TODO: Numerically estimate the Hessian matrix
    raise NotImplementedError


# 5. Integral Calculus
# Task: Compute numerical integrals using Riemann sum and compare with scipy

def numerical_integration():
    """
    Returns:
        float: estimated integral value over [0, 1] for a given function using Riemann sum
    """
    # TODO: Implement Riemann sum or trapezoidal rule
    raise NotImplementedError


# 6. Random Variables & Distributions
# Task: Sample from distributions and compute sample statistics

def distribution_statistics():
    """
    Returns:
        tuple: (sample_mean, sample_variance) of sampled data
    """
    # TODO: Sample from normal, binomial, poisson; compute mean/variance
    raise NotImplementedError


# 7. Naive Bayes Classifier
# Task: Implement and evaluate a Naive Bayes classifier

def naive_bayes_classifier(X_train, y_train, X_test):
    """
    Args:
        X_train (list or np.ndarray): training feature vectors
        y_train (list or np.ndarray): training labels
        X_test (list or np.ndarray): test feature vectors

    Returns:
        list: predicted labels for X_test
    """
    # TODO: Implement a simple Naive Bayes classifier
    raise NotImplementedError


# 8. Statistics
# Task: Compute confidence intervals and perform hypothesis testing

def compute_confidence_interval(data):
    """
    Args:
        data (list or np.ndarray): sample data

    Returns:
        tuple: (sample mean, 95% confidence interval as (lower, upper))
    """
    # TODO: Compute 95% confidence interval for the mean
    raise NotImplementedError


def t_test(sample1, sample2):
    """
    Args:
        sample1 (list or np.ndarray): first sample
        sample2 (list or np.ndarray): second sample

    Returns:
        float: p-value of two-sample t-test
    """
    # TODO: Perform a two-sample t-test
    raise NotImplementedError


# 9. Information Theory
# Task: Compute entropy, cross-entropy, and KL divergence

def information_theory_metrics(p, q):
    """
    Args:
        p (list or np.ndarray): true distribution
        q (list or np.ndarray): predicted distribution

    Returns:
        tuple: (entropy of p, cross-entropy H(p, q), KL divergence D_KL(p || q))
    """
    # TODO: Compute entropy of p, cross-entropy H(p, q), and KL divergence D_KL(p || q)
    raise NotImplementedError


import math
import scipy.stats as stats

# Main execution block to test all exercises with expected outputs
if __name__ == "__main__":
    try:
        assert callable(vector_operations)
        dot, norm_u, norm_v, proj, angle = vector_operations()
        assert isinstance(dot, float)
        assert isinstance(norm_u, float)
        assert isinstance(norm_v, float)
        assert proj.shape == (2,)
        assert isinstance(angle, float)
        print("[✓] Geometry & Linear Algebra")
    except Exception as e:
        print("[✗] Geometry & Linear Algebra:", e)

    try:
        assert callable(eigen_decomposition)
        vals, vecs = eigen_decomposition()
        assert np.allclose(np.dot(vecs, np.dot(np.diag(vals), vecs.T)), np.dot(vecs, np.dot(np.diag(vals), vecs.T)))
        print("[✓] Eigendecomposition")
    except Exception as e:
        print("[✗] Eigendecomposition:", e)

    try:
        assert callable(numerical_derivative)
        result = numerical_derivative(lambda x: x**2, 3.0)
        assert math.isclose(result, 6.0, rel_tol=1e-2)
        print("[✓] Single-variable Calculus")
    except Exception as e:
        print("[✗] Single-variable Calculus:", e)

    try:
        assert callable(gradient_estimation)
        grad = gradient_estimation(lambda xy: xy[0]**2 * xy[1] + np.sin(xy[0] * xy[1]), [1.0, 1.0])
        assert isinstance(grad, np.ndarray) and grad.shape == (2,)
        print("[✓] Multivariable Calculus")
    except Exception as e:
        print("[✗] Multivariable Calculus:", e)

    try:
        assert callable(numerical_integration)
        val = numerical_integration()
        assert isinstance(val, float)
        print("[✓] Integral Calculus")
    except Exception as e:
        print("[✗] Integral Calculus:", e)

    try:
        assert callable(distribution_statistics)
        mean, var = distribution_statistics()
        assert isinstance(mean, float)
        assert isinstance(var, float)
        print("[✓] Random Variables & Distributions")
    except Exception as e:
        print("[✗] Random Variables & Distributions:", e)

    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
        preds = naive_bayes_classifier(X_train, y_train, X_test)
        assert isinstance(preds, list)
        assert len(preds) == len(X_test)
        print("[✓] Naive Bayes Classifier")
    except Exception as e:
        print("[✗] Naive Bayes Classifier:", e)

    try:
        assert callable(compute_confidence_interval)
        mean, ci = compute_confidence_interval([1, 2, 3, 4, 5])
        assert isinstance(ci, tuple) and len(ci) == 2
        assert ci[0] <= mean <= ci[1]
        print("[✓] Statistics")
    except Exception as e:
        print("[✗] Statistics:", e)

    try:
        assert callable(information_theory_metrics)
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        entropy, cross_entropy, kl = information_theory_metrics(p, q)
        assert isinstance(entropy, float)
        assert isinstance(cross_entropy, float)
        assert isinstance(kl, float)
        print("[✓] Information Theory")
    except Exception as e:
        print("[✗] Information Theory:", e)
    
    try:
        assert callable(vector_operations)
        dot, norm_u, norm_v, proj, angle = vector_operations()
        assert isinstance(dot, float)
        assert isinstance(norm_u, float)
        assert isinstance(norm_v, float)
        assert proj.shape == (2,)
        assert isinstance(angle, float)
        print("[✓] Geometry & Linear Algebra")
    except Exception as e:
        print("[✗] Geometry & Linear Algebra:", e)

    try:
        assert callable(eigen_decomposition)
        vals, vecs = eigen_decomposition()
        assert np.allclose(np.dot(vecs, np.dot(np.diag(vals), vecs.T)), np.dot(vecs, np.dot(np.diag(vals), vecs.T)))
        print("[✓] Eigendecomposition")
    except Exception as e:
        print("[✗] Eigendecomposition:", e)

    try:
        assert callable(numerical_derivative)
        result = numerical_derivative(lambda x: x**2, 3.0)
        assert math.isclose(result, 6.0, rel_tol=1e-2)
        print("[✓] Single-variable Calculus")
    except Exception as e:
        print("[✗] Single-variable Calculus:", e)

    try:
        assert callable(gradient_estimation)
        grad = gradient_estimation(lambda xy: xy[0]**2 * xy[1] + np.sin(xy[0] * xy[1]), [1.0, 1.0])
        assert isinstance(grad, np.ndarray) and grad.shape == (2,)
        print("[✓] Multivariable Calculus")
    except Exception as e:
        print("[✗] Multivariable Calculus:", e)

    try:
        assert callable(numerical_integration)
        val = numerical_integration()
        assert isinstance(val, float)
        print("[✓] Integral Calculus")
    except Exception as e:
        print("[✗] Integral Calculus:", e)

    try:
        assert callable(distribution_statistics)
        mean, var = distribution_statistics()
        assert isinstance(mean, float)
        assert isinstance(var, float)
        print("[✓] Random Variables & Distributions")
    except Exception as e:
        print("[✗] Random Variables & Distributions:", e)

    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
        preds = naive_bayes_classifier(X_train, y_train, X_test)
        assert isinstance(preds, list)
        assert len(preds) == len(X_test)
        print("[✓] Naive Bayes Classifier")
    except Exception as e:
        print("[✗] Naive Bayes Classifier:", e)

    try:
        assert callable(compute_confidence_interval)
        mean, ci = compute_confidence_interval([1, 2, 3, 4, 5])
        assert isinstance(ci, tuple) and len(ci) == 2
        assert ci[0] <= mean <= ci[1]
        print("[✓] Statistics")
    except Exception as e:
        print("[✗] Statistics:", e)

    try:
        assert callable(information_theory_metrics)
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        entropy, cross_entropy, kl = information_theory_metrics(p, q)
        assert isinstance(entropy, float)
        assert isinstance(cross_entropy, float)
        assert isinstance(kl, float)
        print("[✓] Information Theory")
    except Exception as e:
        print("[✗] Information Theory:", e)
