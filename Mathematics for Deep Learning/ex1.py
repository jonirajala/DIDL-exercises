# Boilerplate Exercise Code for Mathematics for Deep Learning (D2L.ai Appendix)

# 1. Geometry & Linear Algebra
# Task: Compute vector norms, dot products, projections, and angles
import numpy as np

# 1. Geometry & Linear Algebra
def vector_operations(u, v):
    """
    Args:
        u (np.ndarray): 2D vector
        v (np.ndarray): 2D vector

    Returns:
        tuple: (dot_product, norm_u, norm_v, projection of u on v, angle between u and v in radians)
    """
    dot_prod = u.dot(v)
    n_u = np.linalg.norm(u)
    n_v = np.linalg.norm(v)
    projection = (dot_prod / n_v**2) * v
    angle = np.acos(dot_prod / (n_u * n_v))

    return dot_prod, n_u, n_v, projection, angle

# 2. Eigendecomposition
def eigen_decomposition(A):
    """
    Args:
        A (np.ndarray): Symmetric matrix

    Returns:
        tuple: (eigenvalues, eigenvectors) of the matrix A
    """
    v, Q = np.linalg.eigh(A)
    return (v, Q)


# 3. Single-variable Calculus
def numerical_derivative(f, x, h=1e-5):
    """
    Args:
        f (callable): function of a single variable
        x (float): point at which to evaluate derivative
        h (float): step size for central difference

    Returns:
        float: estimated derivative at x
    """
    # raise NotImplementedError("This function has not been implemented yet.")
#    np.linalg.
    return (f(x+h) - f(x)) / h


# 4. Multivariable Calculus

def gradient_estimation(f, point, h=1e-5):
    """
    Args:
        f (callable): function taking a list [x, y]
        point (list): [x, y] at which to compute gradient
        h (float): step size

    Returns:
        np.ndarray: estimated gradient vector
    """
    # raise NotImplementedError("This function has not been implemented yet.")
    x, y = point[0], point[1]
    dx = (f((x+h, y)) - f((x-h,y))) / (2*h)
    dy = (f((x, y+h)) - f((x,y-h))) / (2*h)

    return np.array([dx, dy])

def hessian_estimation(f, point, h=1e-5):
    """
    Args:
        f (callable): function taking a list [x, y]
        point (list): [x, y] at which to compute Hessian
        h (float): step size

    Returns:
        np.ndarray: 2x2 Hessian matrix
    """
    # raise NotImplementedError("This function has not been implemented yet.")
    x, y = point

    f_xx = (f([x + h, y]) - 2 * f([x, y]) + f([x - h, y])) / (h ** 2)
    f_yy = (f([x, y + h]) - 2 * f([x, y]) + f([x, y - h])) / (h ** 2)
    f_xy = (f([x + h, y + h]) - f([x + h, y - h])
            - f([x - h, y + h]) + f([x - h, y - h])) / (4 * h ** 2)

    return np.array([[f_xx, f_xy],
                     [f_xy, f_yy]])

# 5. Integral Calculus
def numerical_integration(f, a, b, n=1000):
    """
    Args:
        f (callable): function to integrate
        a (float): lower bound
        b (float): upper bound
        n (int): number of intervals

    Returns:
        float: estimated integral value
    """
    # raise NotImplementedError("This function has not been implemented yet.")
    step = ((b-a) / n)
    ret = 0.5 * (f(a) + f(b)) * step
    for i in range(1, n):
        ret += f(a+(step * i)) * step
    return ret


# 6. Random Variables & Distributions
def distribution_statistics(samples):
    """
    Args:
        samples (np.ndarray): data samples

    Returns:
        tuple: (sample_mean, sample_variance)
    """
    return (samples.mean(), np.var(samples, ddof=1))


# 7. Naive Bayes Classifier
def naive_bayes_classifier(X_train, y_train, X_test):
    """
    Args:
        X_train (np.ndarray): training feature vectors
        y_train (np.ndarray): training labels
        X_test (np.ndarray): test feature vectors

    Returns:
        list: predicted labels for X_test
    """
    n_y = np.bincount(y_train)
    P_y = n_y / n_y.sum()
    print(P_y)
    print(X_train.shape)
    n_x = np.zeros((len(n_y), X_train.shape[1]))
    for y in range(len(n_y)):
        n_x[y] += np.array(X_train[y_train == y].sum(axis=0))
    P_xy = (n_x + 1) / (n_y + 2).reshape(3, 1)

    log_P_xy = np.log(P_xy)
    log_P_xy_neg = np.log(1 - P_xy)
    log_P_y = np.log(P_y)

    def bayes_pred_stable(x):
        p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
        p_xy = p_xy.reshape(3, -1).sum(axis=1)  # p(x|y)
        return p_xy + log_P_y

    return [bayes_pred_stable(x).argmax().item()
            for x in X_test]

# 8. Statistics
def compute_confidence_interval(data):
    """
    Args:
        data (list or np.ndarray): sample data

    Returns:
        tuple: (sample mean, 95% confidence interval as (lower, upper))
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    se = std / np.sqrt(n)

    confidence = 0.95
    z = stats.norm.ppf((1 + confidence) / 2)  # z = 1.96
    lower = mean - z * se
    upper = mean + z * se

    return (mean, (lower, upper))

    


def t_test(sample1, sample2):
    """
    Args:
        sample1 (list or np.ndarray): first sample
        sample2 (list or np.ndarray): second sample

    Returns:
        float: p-value of two-sample t-test
    """
    t, p = stats.ttest_ind(sample1, sample2)
    return p


# 9. Information Theory
def information_theory_metrics(p, q):
    """
    Args:
        p (list or np.ndarray): true distribution
        q (list or np.ndarray): predicted distribution

    Returns:
        tuple: (entropy of p, cross-entropy H(p, q), KL divergence D_KL(p || q))
    """
    raise NotImplementedError("This function has not been implemented yet.")


import numpy as np
import math
import scipy.stats as stats

if __name__ == "__main__":
    try:
        u = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        dot, norm_u, norm_v, proj, angle = vector_operations(u, v)

        # Manual checks using numpy
        dot_check = np.dot(u, v)
        norm_u_check = np.linalg.norm(u)
        norm_v_check = np.linalg.norm(v)
        proj_check = (dot_check / norm_v_check**2) * v
        angle_check = np.arccos(dot_check / (norm_u_check * norm_v_check))

        assert np.isclose(dot, dot_check), "Dot product incorrect"
        assert np.isclose(norm_u, norm_u_check), "Norm of u incorrect"
        assert np.isclose(norm_v, norm_v_check), "Norm of v incorrect"
        assert np.allclose(proj, proj_check), "Projection incorrect"
        assert np.isclose(angle, angle_check), "Angle incorrect"
        print("[✓] Geometry & Linear Algebra")
    except Exception as e:
        print("[✗] Geometry & Linear Algebra:", e)

    try:
        A = np.array([[2, -1], [-1, 2]])
        vals, vecs = eigen_decomposition(A)
        # Check if reconstruction works
        A_reconstructed = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
        assert np.allclose(A, A_reconstructed), "Eigendecomposition reconstruction failed"
        print("[✓] Eigendecomposition")
    except NotImplementedError:
        print("[✗] Eigendecomposition: Not Implemented")
    except Exception as e:
        print("[✗] Eigendecomposition:", e)

    try:
        f = lambda x: x**2
        x = 3.0
        result = numerical_derivative(f, x)
        analytical = 2 * x
        assert np.isclose(result, analytical, atol=1e-3), f"Numerical derivative incorrect, {result} - {analytical}"
        print("[✓] Single-variable Calculus")
    except NotImplementedError:
        print("[✗] Single-variable Calculus: Not Implemented")
    except Exception as e:
        print("[✗] Single-variable Calculus:", e)

    try:
        f2 = lambda xy: xy[0]**2 * xy[1] + np.sin(xy[0] * xy[1])
        point = [1.0, 1.0]
        grad = gradient_estimation(f2, point)

        # Approximate gradient using central difference
        eps = 1e-5
        dx = (f2([point[0] + eps, point[1]]) - f2([point[0] - eps, point[1]])) / (2 * eps)
        dy = (f2([point[0], point[1] + eps]) - f2([point[0], point[1] - eps])) / (2 * eps)
        grad_check = np.array([dx, dy])
        assert np.allclose(grad, grad_check, atol=1e-5), f"Gradient incorrect, {grad} - {grad_check}"
        print("[✓] Multivariable Calculus: Gradient")
    except NotImplementedError:
        print("[✗] Multivariable Calculus: Gradient Not Implemented")
    except Exception as e:
        print("[✗] Multivariable Calculus: Gradient", e)

    try:
        hess = hessian_estimation(f2, point)

        # Approximate Hessian using finite difference (second-order)
        eps = 1e-5
        f_xx = (f2([point[0] + eps, point[1]]) - 2 * f2(point) + f2([point[0] - eps, point[1]])) / eps**2
        f_yy = (f2([point[0], point[1] + eps]) - 2 * f2(point) + f2([point[0], point[1] - eps])) / eps**2
        f_xy = (f2([point[0] + eps, point[1] + eps]) - f2([point[0] + eps, point[1] - eps])
                - f2([point[0] - eps, point[1] + eps]) + f2([point[0] - eps, point[1] - eps])) / (4 * eps**2)
        hess_check = np.array([[f_xx, f_xy], [f_xy, f_yy]])

        assert np.allclose(hess, hess_check, atol=1e-3), f"Hessian incorrect, {hess}, {hess_check}"
        print("[✓] Multivariable Calculus: Hessian")
    except NotImplementedError:
        print("[✗] Multivariable Calculus: Hessian Not Implemented")
    except Exception as e:
        print("[✗] Multivariable Calculus: Hessian", e)

    try:
        integral_result = numerical_integration(lambda x: x**2, 0.0, 1.0, n=1000)
        expected = 1.0 / 3.0
        assert np.isclose(integral_result, expected, atol=1e-4), "Integral result incorrect"
        print("[✓] Integral Calculus")
    except NotImplementedError:
        print("[✗] Integral Calculus: Not Implemented")
    except Exception as e:
        print("[✗] Integral Calculus:", e)

    try:
        samples = np.random.normal(loc=0, scale=1, size=1000)
        mean, var = distribution_statistics(samples)
        assert np.isclose(mean, np.mean(samples), atol=1e-6), "Sample mean incorrect"
        assert np.isclose(var, np.var(samples, ddof=1), atol=1e-6), "Sample variance incorrect"
        print("[✓] Random Variables & Distributions")
    except NotImplementedError:
        print("[✗] Random Variables & Distributions: Not Implemented")
    except Exception as e:
        print("[✗] Random Variables & Distributions:", e)

    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
        preds = naive_bayes_classifier(X_train, y_train, X_test)
        assert isinstance(preds, list) and len(preds) == len(X_test), "Naive Bayes predictions invalid"
        print("[✓] Naive Bayes Classifier")
    except NotImplementedError:
        print("[✗] Naive Bayes Classifier: Not Implemented")
    except Exception as e:
        print("[✗] Naive Bayes Classifier:", e)

    try:
        data = [1, 2, 3, 4, 5]
        mean, ci = compute_confidence_interval(data)
        assert np.isclose(mean, np.mean(data)), "Mean incorrect"
        assert isinstance(ci, tuple) and len(ci) == 2, "CI format incorrect"
        assert ci[0] <= mean <= ci[1], "Mean not in CI"
        print("[✓] Statistics: Confidence Interval")
    except NotImplementedError:
        print("[✗] Statistics: Confidence Interval Not Implemented")
    except Exception as e:
        print("[✗] Statistics: Confidence Interval", e)

    try:
        s1 = [1, 2, 3, 4, 5]
        s2 = [2, 3, 4, 5, 6]
        p_val = t_test(s1, s2)
        assert 0.0 <= p_val <= 1.0, "p-value should be between 0 and 1"
        print("[✓] Statistics: T-test")
    except NotImplementedError:
        print("[✗] Statistics: T-test Not Implemented")
    except Exception as e:
        print("[✗] Statistics: T-test", e)

    try:
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        entropy, cross_entropy, kl = information_theory_metrics(p, q)

        def entropy_func(p):
            return -np.sum(p * np.log2(p))

        def cross_entropy_func(p, q):
            return -np.sum(p * np.log2(q))

        def kl_div(p, q):
            return np.sum(p * np.log2(p / q))

        assert np.isclose(entropy, entropy_func(p)), "Entropy incorrect"
        assert np.isclose(cross_entropy, cross_entropy_func(p, q)), "Cross-entropy incorrect"
        assert np.isclose(kl, kl_div(p, q)), "KL divergence incorrect"
        print("[✓] Information Theory")
    except NotImplementedError:
        print("[✗] Information Theory: Not Implemented")
    except Exception as e:
        print("[✗] Information Theory:", e)
