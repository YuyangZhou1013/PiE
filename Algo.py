import numpy as np
import warnings

warnings.filterwarnings("ignore")


def ISTA(A, b, mu, shrinkage, a, Lambda, maxiter, error): 
    x0 = np.zeros((A.shape[1], 1))
    aa = np.eye(A.shape[1]) - (mu * A.T.dot(A))
    ab = mu * A.T.dot(b)
    for i in range(maxiter):  
        x1 = aa.dot(x0) + ab
        x = shrinkage(x1, a, mu*Lambda)
        if (np.linalg.norm(x - x0, 2) / (np.linalg.norm(x) + 1)) < error:
            break
        x0 = x.copy()
    return x, i
