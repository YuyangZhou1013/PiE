import numpy as np
from scipy.special import lambertw
import warnings
import math
warnings.filterwarnings("ignore")


# log shrinkage
def r(z, Lambda, epsilon):
    return q(Lambda, z, r2(z, epsilon, Lambda), epsilon) - q(Lambda, z, 0, epsilon)


def bisection_lg(a, b, Lambda, epsilon):
    for i in range(30):
        c = (a + b) / 2
        if r(a, Lambda, epsilon) * r(c, Lambda, epsilon) < 0:
            b = c
        else:
            a = c
        if (b - a) < 1e-8:
            break
    return c


def r2(x, epsilon, Lambda):
    return 0.5 * (x - epsilon) + np.sqrt(0.25 * (x + epsilon) ** 2 - Lambda)


def q(Lambda, z, x, epsilon):
    return (1 / (2 * Lambda)) * (x - z) ** 2 + np.log(1 + abs(x) / epsilon)


def z(epsilon, Lambda):
    return bisection_lg(2 * np.sqrt(Lambda) - epsilon, Lambda / epsilon, Lambda, epsilon)


def shrinkage_log_soft(x0, epsilon, Lambda):
    x = np.zeros((x0.shape[0], 1))

    index1 = abs(x0) > Lambda / epsilon
    x[index1] = np.sign(x0[index1]) * r2(abs(x0[index1]), epsilon, Lambda)
    return x


def shrinkage_log_hard(x0, epsilon, Lambda):
    z_ = z(epsilon, Lambda)
    x = np.zeros((x0.shape[0], 1))
    index1 = abs(x0) > z_
    x[index1] = np.sign(x0[index1]) * r2(abs(x0[index1]), epsilon, Lambda)
    return x


def shrinkage_log(a_log, Lambda):
    if np.sqrt(Lambda) <= a_log:
        return shrinkage_log_soft
    else:
        return shrinkage_log_hard


# TL1 shrinkage
def tau(x0, a, Lambda):  # tl1 shrinkage
    if Lambda < a / 2:
        t = a * Lambda / (a + 1)
    else:
        t = (Lambda + a / 2) ** 2 / (2 * (a + 1))
    return t


def varphi(x0, a, Lambda):
    return np.arccos(1 - ((27 * tau(x0, a, Lambda) * a * (a + 1)) / (2 * ((a + abs(x0)) ** 3))))


def shrinkage_TL1(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index0 = abs(x0) > Lambda

    x[index0] = np.sign(x0[index0]) * (
            (2 / 3) * (a + abs(x0[index0])) * np.cos(varphi(x0[index0], a, Lambda) / 3) - (2 * a) / 3 + abs(
        x0[index0]) / 3)
    return x


# scad shrinkage
def shrinkage_scad(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index0 = abs(x0) <= Lambda
    index1 = np.logical_and(Lambda < abs(x0), abs(x0) <= 2 * Lambda)
    index2 = np.logical_and(2 * Lambda < abs(x0), abs(x0) <= a * Lambda)
    index3 = abs(x0) > a * Lambda

    x[index0] = 0
    x[index1] = np.sign(x0[index1]) * (abs(x0[index1]) - Lambda)
    x[index2] = ((a - 1) * x0[index2] - np.sign(x0[index2]) * a * Lambda) / (a - 2)
    x[index3] = x0[index3]
    return x


# pie shrinkage
def P(z, mu_Lambda, sigma):
    return 0.5 + mu_Lambda * ((z / sigma + 1) * np.exp(-z / sigma) - 1) / (z ** 2)


def bisection_PiE(a, b, mu_Lambda, sigma):
    for i in range(30):
        c = (a + b) / 2
        if P(a, mu_Lambda, sigma) * P(c, mu_Lambda, sigma) < 0:
            b = c
        else:
            a = c
        if (b - a) < 1e-8:
            break
    return c


def PiEProximalbyLambertWThreshold(x, sigma, mu_Lambda):
    n = np.shape(x)[0]
    y = np.zeros((n, 1))
    xstar = bisection_PiE(0, np.sqrt(2 * mu_Lambda), mu_Lambda, sigma)
    Threshold = xstar + mu_Lambda * np.exp(-xstar / sigma)

    index = abs(x) > Threshold

    z = -(mu_Lambda / sigma ** 2) * np.exp(-abs(x[index]) / sigma)

    lamb = lambertw(z, 0).real

    y[index] = np.sign(x[index]) * (sigma * lamb + abs(x[index]))

    return y


def shrinkage_PiE_soft(x0, sigma, mu_Lambda):
    x = np.zeros((np.shape(x0)[0], 1))
    index = abs(x0) > mu_Lambda / sigma

    z = -(mu_Lambda / sigma ** 2) * np.exp(-abs(x0[index]) / sigma)
    x1 = sigma * lambertw(z, 0).real + abs(x0[index])
    x[index] = np.sign(x0[index]) * x1
    return x



# mcp shrinkage
def shrinkage_mcp(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))

    index1 = np.logical_and(abs(x0) > Lambda, abs(x0) <= a * Lambda)
    index2 = abs(x0) > a * Lambda

    x[index1] = np.sign(x0[index1]) * (abs(x0[index1]) - Lambda) / (1 - 1 / a)
    x[index2] = x0[index2]

    return x


# cap shrinkage
def shrinkage_CaP_hard(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index = abs(x0) > np.sqrt(2 * a * Lambda)
    x[index] = x0[index]
    return x


def shrinkage_CaP_soft(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index1 = abs(x0) > a + Lambda / 2
    index2 = np.logical_and(abs(x0) > Lambda, abs(x0) <= a + Lambda / 2)

    x[index1] = x0[index1]
    x[index2] = np.sign(x0[index2]) * (abs(x0[index2]) - Lambda)
    return x


def shrinkage_CaP(a_cap, Lambda):
    if Lambda <= 2 * a_cap:
        return shrinkage_CaP_soft
    else:
        return shrinkage_CaP_hard


# half shrinkage
def shrinkage_half(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))

    index = abs(x0) > Lambda

    theta = np.arccos(np.sqrt(2) / 2 * np.power(Lambda / abs(x0[index]), 3 / 2))

    x[index] = (2 / 3) * x0[index] * (1 + np.cos(2 * np.pi / 3 - (2 * theta) / 3))

    return x


# hard shrinkage
def shrinkage_hard(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))

    index = abs(x0) > Lambda

    x[index] = x0[index]

    return x


# soft shrinkage
def shrinkage_soft(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))

    index = abs(x0) > Lambda

    x[index] = np.sign(x0[index]) * (abs(x0[index]) - Lambda)

    return x



def shrinkage_l1_2(x0, a, Lambda):

    infnorm = np.linalg.norm(x0, ord=np.inf) 
    n = np.shape(x0)[0]
    x = np.zeros((n, 1))

    if infnorm > Lambda:
        temp = shrinkage_soft(x0, 0, Lambda)
        x = temp * (np.linalg.norm(temp) + a * Lambda) / np.linalg.norm(temp)
    elif infnorm == Lambda:
        index = abs(x0) == Lambda
        x[index] = np.sign(x0[index]) * (a * Lambda)
    elif np.logical_and(infnorm > ((1 - a) * Lambda), infnorm < Lambda):
        index = np.argmax(abs(x0))
        x[index] = np.sign(x0[index]) * (infnorm + (a - 1) * Lambda)

    return x


def getR(rho,qt,nq,t0,maxiter):
    phat = -np.linalg.norm(qt)**2/3
    b = np.sqrt(abs(phat))
    r = (4*nq/rho)**(1/3)
    flag = 0
    for i in range(maxiter):
        rt = r
        delta = rho*nq-t0/r
        Del = rho**2 - 4*(delta)/r**3
        if Del < 0 :
            flag = 1
            break
        else:
            a = r**3*(rho-np.sqrt(Del))/2
            qhat = (nq-a)/(2*rho)
            phi = np.arccos(qhat/b**3)
            r = 2*b*np.cos(math.pi/3-phi/3)
        if abs(r-rt)/abs(r)<1e-3:
            break
    return r,flag

def shrinkage_l1overl2(qq,a,Lambda):
    rho = 1/Lambda
    signQ = np.sign(qq)
    qq = abs(qq)
    index = np.argsort(-qq,axis=0)
    q = qq[index,0]
    N = q.shape[0]
    t1 = 1
    t2 = N
    a = None
    r = None
    nnz = None
    aa = np.zeros((N,1))
    X = np.zeros((N,1))
    x = np.zeros((N,1))
    maxiter = 500
    t0 = int((t1+t2)/2)
    flag = 0
    flag3 = 0
    if rho > (1/q[0]**2):
        flag3 = 1
    else:
        aa[index[0]] = q[0]
        x = aa*signQ #onesp
    while((t2-t1)>1 and flag3):
        qt = q[:t0]
        nq = np.sum(abs(qt))
        r_,flag = getR(rho,qt,nq,t0,maxiter)
        r = abs(r_)
        if flag == 1:
            if (q[t0]-1/(rho*r))>0:
                t1 = t0
            else:
                t2 = t0
        else:
            r = r_
            a = r**3*(rho-np.sqrt(rho**2-4*(rho*nq-t0/r)/(r**3)))/2
            flag2 = (q[t0-1]-1/(rho*r))*(1-a/(rho*r**3))
            if flag2 > 0:
                if (q[t0]-1/(rho*r))*(1-a/(rho*r**3)) <= 0:
                    numb = 1-a/(rho*r**3)
                    xx = (qt-1/(rho*r))/numb
                    aa[:t0] = xx
                    X[index,0] = aa
                    x = X*signQ
                    return x
                else:
                    t1 = t0+1
            else:
                t2 = t0
        t0 = int((t1+t2)/2)
    if flag3:
        if t0 == 1:
            aa[index[0]] = q[0]
            x = aa*signQ
        else:
            qt = q[:t0]
            q2n = np.sum(qt**2)
            nq = np.sum(abs(qt))
            r_,flag_ = getR(rho,qt,nq,t0,maxiter)
            r = r_
            a = nq-rho*(q2n*r-r**3)
            numb = 1-a/(rho*r**3)
            xx = (qt-1/(rho*r))/numb
            aa[:t0] = xx
            X[index,0] = aa
            x = X*signQ
    return x