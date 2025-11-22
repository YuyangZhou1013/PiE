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
def tau(a, Lambda):  # tl1 shrinkage
    if Lambda <= a**2 / (2 * (a + 1)):
        t = Lambda * (a + 1) / a
    else:
        t = np.sqrt(2 * Lambda * (a+1))- a / 2
    return t

def varphi(x0, a, Lambda):
    return np.arccos(1 - 27 * Lambda * a * (a + 1) / (2 * (a + abs(x0)) ** 3))

def shrinkage_TL1(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    t= tau(a, Lambda)
    index0 = abs(x0) > t
    abs_x0 = abs(x0[index0])
    varphi_val = varphi(x0[index0], a, Lambda)
    x[index0] = np.sign(x0[index0]) * (
            (2 / 3) * (a + abs_x0) * np.cos(varphi_val / 3) - 2 * a / 3 + abs_x0 / 3)
    return x


# scad shrinkage
#此处nu和a是形状参数，Lambda是临近算子参数
def shrinkage_scad(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    nu = 1
    index1 = np.logical_and(Lambda*nu < np.abs(x0), np.abs(x0) <= nu*(Lambda + 1))
    index2 = np.logical_and(nu*(Lambda + 1) < np.abs(x0), np.abs(x0) <= a*nu)
    index3 = np.abs(x0) > a*nu
    x[index1] = x0[index1] - nu*Lambda*np.sign(x0[index1])
    x[index2] = ((a-1)*abs(x0[index2]) - a*nu*Lambda)*np.sign(x0[index2])/(a-1-Lambda)
    x[index3] = x0[index3]
    return x

# # mcp shrinkage
# def shrinkage_mcp(x0, a, Lambda):
#     x = np.zeros((x0.shape[0], 1))
#     index1 = np.logical_and(abs(x0) > Lambda, abs(x0) <= a * Lambda)
#     index2 = abs(x0) > a * Lambda
#     x[index1] = np.sign(x0[index1]) * (abs(x0[index1]) - Lambda) / (1 - 1 / a)
#     x[index2] = x0[index2]
#     return x

# mcp shrinkage
def shrinkage_mcp(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index1 = np.logical_and(abs(x0) > Lambda, abs(x0) <= a)
    index2 = abs(x0) > a
    x[index1] = np.sign(x0[index1]) * (abs(x0[index1]) - Lambda) * a / (a - Lambda)
    x[index2] = x0[index2]
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


# CL1 shrinkage
def shrinkage_CaP1(x0, Lambda, nu):
    x = np.zeros((x0.shape[0], 1))
    index0 = np.logical_and(nu/Lambda<abs(x0),abs(x0) <= Lambda+nu/(2*Lambda))
    index1 = abs(x0) > Lambda+nu/(2*Lambda)
    x[index0] = x0[index0]-np.sign(x0[index0])*nu/Lambda
    x[index1] = x0[index1]
    return x

#CL1/2 shrinkage
def J(q,nu,Lambda,u,tau):
    return nu*np.minimum(1.0,np.power(np.abs(u)/Lambda,q)) + 0.5 * (u -tau)**2

def ProxL1over2(nu, tau):
    ytilde = np.zeros_like(tau)
    condition = np.abs(tau) > 1.5 * np.power(nu, 2 / 3)
    theta = np.arccos((-np.power(3, 3 / 2) / 4) * nu * np.power(abs(tau[condition]), -3 / 2))
    ytilde[condition] = (2 / 3) * tau[condition] * (1 + np.cos((2 * theta) / 3))
    return ytilde

def bisection_CL1over2(a,b,q,nu,Lambda):
    for i in range(20):
        c=(a+b)/2
        if (J(q,nu,Lambda,ProxL1over2(nu/Lambda**q,a),a)-nu) * (J(q,nu,Lambda,ProxL1over2(nu/Lambda**q,c),c)-nu)<0:
            b=c
        else:
            a=c
        if (b-a)<1e-5:
            break
    return c

def shrinkage_CaP1over2(x0,Lambda,nu):
    x = np.zeros((np.shape(x0)[0], 1))
    cnulambda_1over2 = (3 / 2) * np.power(nu / np.power(Lambda, 1 / 2), 2 / 3)
    C_nulambda_1over2 = bisection_CL1over2(cnulambda_1over2, Lambda + (1 / 2) * nu / Lambda, 1 / 2, nu, Lambda)
    index0 = abs(x0) < C_nulambda_1over2
    index1 = abs(x0) >= C_nulambda_1over2
    x[index0] = ProxL1over2(nu/np.power(Lambda,1/2),x0[index0])
    x[index1] = x0[index1]
    return x

#CL2/3 shrinkage
def ProxL2over3(nu,tau):
    ytilde = np.zeros_like(tau)
    condition = np.abs(tau) > 2*np.power(2*nu/3,3/4)
    t1 = np.sqrt(tau[condition] ** 4 / 256 - 8 * (nu ** 3) / 729)
    t = 2 * (np.power(tau[condition] ** 2 / 16 + t1, 1 / 3) + np.power(tau[condition] ** 2 / 16 - t1, 1 / 3))
    ytilde[condition]=(1/8)*np.sign(tau[condition])*np.power(np.sqrt(t)+np.sqrt(2*np.abs(tau[condition])/np.sqrt(t) -t),3)
    return ytilde

def bisection_CL2over3(a,b,q,nu,Lambda):
    for i in range(20):
        c=(a+b)/2
        if (J(q,nu,Lambda,ProxL2over3(nu/Lambda**q,a),a)-nu)*(J(q,nu,Lambda,ProxL2over3(nu/Lambda**q,c),c)-nu)<0:
            b=c
        else:
            a=c
        if (b-a)<1e-5:
            break
    return c

def shrinkage_CaP2over3(x0,Lambda,nu):
    x = np.zeros((np.shape(x0)[0], 1))
    cnulambda_2over3=2*np.power((2/3)*nu/np.power(Lambda,2/3),3/4)
    C_nulambda_2over3 = bisection_CL2over3(cnulambda_2over3, Lambda + (2 / 3) * nu / Lambda, 2 / 3, nu, Lambda)
    index0 = abs(x0) < C_nulambda_2over3
    index1 = abs(x0) >= C_nulambda_2over3
    x[index0] = ProxL2over3(nu / np.power(Lambda, 2 / 3), x0[index0])
    x[index1] = x0[index1]
    return x

#half shrinkage
def  shrinkage_half(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    threshold = (3 / 2) * np.power(Lambda, 2 / 3)
    index = abs(x0) > threshold
    theta = np.arccos(- np.power(3, 3/2) / 4 * Lambda * np.power(abs(x0[index]), -3/2))
    x[index] = (2 / 3) * x0[index] * (1 + np.cos((2 * theta) / 3))
    return x

# l2/3 shrinkage
def shrinkage_l2over3(x0,a,Lambda):
    x = np.zeros((x0.shape[0], 1))
    threshold = 2 * np.power(2 * Lambda / 3, 3/4)
    index = abs(x0) > threshold
    x0_index = x0[index]
    term1 = np.power(x0_index,2)/16 + np.sqrt(np.power(x0_index,4)/256 - 8*np.power(Lambda,3)/729)
    term2 = np.power(x0_index,2)/16 - np.sqrt(np.power(x0_index,4)/256 - 8*np.power(Lambda,3)/729)
    s = np.power(term1,1/3) + np.power(term2,1/3)
    theta = np.sqrt(2*s) + np.sqrt(2*abs(x0_index) / np.sqrt(2*s) - 2*s)
    x[index] = np.sign(x0_index)*np.power(theta,3)/8
    return x

# soft shrinkage
def shrinkage_soft(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index = abs(x0) > Lambda
    x[index] = np.sign(x0[index]) * (abs(x0[index]) - Lambda)
    return x

# hard shrinkage
def shrinkage_hard(x0, a, Lambda):
    x = np.zeros((x0.shape[0], 1))
    index = abs(x0) > np.sqrt(2*Lambda)
    x[index] = x0[index]
    return x


# l1-l2 shrinkage
def shrinkage_l1_2(x0, a, Lambda):
    infnorm = np.linalg.norm(x0, ord=np.inf) 
    n = np.shape(x0)[0]
    x = np.zeros((n, 1))
    if infnorm > Lambda:
        temp = shrinkage_l1(x0, 0, Lambda)
        x = temp * (np.linalg.norm(temp) + a * Lambda) / np.linalg.norm(temp)
    elif infnorm == Lambda:
        index = abs(x0) == Lambda
        x[index] = np.sign(x0[index]) * (a * Lambda)
    elif np.logical_and(infnorm > ((1 - a) * Lambda), infnorm < Lambda):
        index = np.argmax(abs(x0))
        x[index] = np.sign(x0[index]) * (infnorm + (a - 1) * Lambda)
    return x


# l1/l2 shrinkage
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