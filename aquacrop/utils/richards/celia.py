import numpy as np
import pandas as pd

def Cfun(psi, pars):
    x3 = (pars['alpha'] + np.abs(psi) ** pars['beta']) ** 2.
    x2 = pars['alpha'] / x3
    x1 = pars['beta'] * np.abs(psi) ** (pars['beta'] - 1) * x2
    C = (pars['thetaS'] - pars['thetaR']) * x1
    return C


def Kfun(psi, pars):
    x2 = pars['A'] + np.abs(psi) ** pars['gamma']
    x1 = pars['A'] / x2
    K = pars['Ks'] * x1
    return K


def thetafun(psi, pars):
    x3 = pars['alpha'] + np.abs(psi) ** pars['beta']
    x2 = pars['alpha'] / x3
    x1 = (pars['thetaS'] - pars['thetaR']) * x2
    theta = pars['thetaR'] + x1
    return theta


def solverfun(R, C, Kmid, dt, dz, n):
    # Initialize arrays
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    y = np.zeros(n)

    # Construct matrix
    a = -Kmid[:-1] * dt / C / dz ** 2.
    b = 1. + (Kmid[:-1] + Kmid[1:]) * dt / C / dz ** 2.
    c = -Kmid[1:] * dt / C / dz ** 2.
    A = np.diag(a[1:], -1) + np.diag(b, 0) + np.diag(c[:-1], 1)

    # Construct RHS
    y[:] = R[:] * dt / C

    # Boundary conditions - nothing to do
    # y[0]=y[0]
    # y[-1]=y[-1]

    # Solve:
    dell = np.linalg.solve(A, y)

    return dell


def Rfun(psiiter, psiin, psiT, psiB, C, Kmid, dt, dz, n):
    # This solves the Picard residual term:
    psigrid = np.hstack([psiB, psiiter, psiT])
    x1 = -C * (psiiter - psiin) / dt
    x2 = 1 / dz ** 2 * (Kmid[1:] * (psigrid[2:] - psigrid[1:-1]) - Kmid[:-1] * (psigrid[1:-1] - psigrid[:-2]))
    x3 = (Kmid[1:] - Kmid[:-1]) / dz
    R = x1 + x2 + x3

    return R


def iterfun(psiin, pars, psiT, psiB, dt, dz, n):
    # psiin = psi^n
    # psiiter = psi^n+1,m
    # psiout = psi^n+1,m+1

    tolerance = 1e-10
    maxcount = 1000
    Rmin = 1.

    # Initialize arrays
    psiiter = np.zeros(len(psiin))
    psiout = np.zeros(len(psiin))

    # Initial guess: psi_n+1^1 = psi_n
    psiiter[:] = psiin[:]

    count = 0.
    while count <= 1 or (Rmin >= tolerance and count <= maxcount):
        # Get C,K:
        C = Cfun(psiiter, pars)
        K = Kfun(np.hstack([psiB, psiiter, psiT]), pars)
        Kmid = (K[1:] + K[:-1]) / 2.
        # Get R
        R = Rfun(psiiter, psiin, psiT, psiB, C, Kmid, dt, dz, n)
        # Solve for del
        dell = solverfun(R, C, Kmid, dt, dz, n)
        # Update psi estimates at different iteration levels
        psiout[:] = psiiter[:] + dell[:]

        #        # Plot to check convergence:
        #        pl.plot(z,psiiter)
        #        pl.plot(z,psiout)
        #        pl.show()

        err = psiout - psiiter
        psiiter[:] = psiout[:]
        Rmin = np.abs(np.min(R))
        count += 1

    # print('Iteration count = %d'%(count-1))

    return psiout


def massbal(psi, psiT, psiB, pars, n, dt, dz):
    # Initial storage:
    theta = thetafun(psi, pars)
    S = np.sum(theta * dz, 1)
    S0 = S[0]
    SN = S[-1]

    # Inflow:
    Kin = (Kfun(psiB, pars) + Kfun(psi[:, 0], pars)) / 2.
    QIN = -Kin * ((psi[:, 0] - psiB) / dz + 1.)
    QIN[0] = 0.
    QINsum = np.sum(QIN) * dt

    # Outflow:
    Kout = (Kfun(psi[:, -1], pars) + Kfun(psiT, pars)) / 2.
    QOUT = -Kout * ((psiT - psi[:, -1]) / dz + 1.)
    QOUT[0] = 0.
    QOUTsum = np.sum(QOUT) * dt

    # Balance:
    dS = SN - S0
    dQ = QINsum - QOUTsum
    err = dS / dQ

    return QIN, QOUT, S, err


def ModelRun(dt, dz, n, nt, psi, psiB, psiT, pars):
    # Solve:
    for j in range(1, nt):
        psi[j, :] = iterfun(psi[j - 1, :], pars, psiT, psiB, dt, dz, n)

    QIN, QOUT, S, err = massbal(psi, psiT, psiB, pars, n, dt, dz)

    return psi, QIN, QOUT, S, err



def setpars():
    pars = {}
    pars['thetaR'] = 0.075
    pars['thetaS'] = 0.287
    pars['alpha'] = 1.611e6
    pars['beta'] = 3.96
    pars['A'] = 1.175e6
    pars['gamma'] = 4.74
    pars['Ks'] = 0.00944
    return pars


def setup(dti):
    # Set parameters:
    pars = setpars()

    # Grid:
    zN = 40.
    dz = 1.
    tN = 360.

    z = np.arange(dz, zN, dz)
    n = len(z)

    t = np.arange(0, tN + dti, dti)
    nt = len(t)

    # Initialize array:
    psi = np.zeros((nt, n))

    # ICs:
    psi[0, :] = -61.5

    # BCs:
    psiB = -61.5
    psiT = -20.7

    return z, t, dz, n, nt, zN, psi, psiB, psiT, pars


def main():
    dt = np.array([0.1, 1, 3, 10, 20, 30, 40, 60, 90, 120])
    for dti in dt:
        z, t, dz, n, nt, zN, psi, psiB, psiT, pars = setup(dti)
        psi, QIN, QOUT, S, err = ModelRun(dti, dz, n, nt, psi, psiB, psiT, pars)
        z = np.hstack([0, z, zN])
        z = z[-1] - z
        psi = np.hstack([psiB, psi[-1, :], psiT])

        # Save data:
        dtstr = str(dti).replace('.', '-')
        fname = 'psi_dt%s.csv' % dtstr
        f = open(fname, 'w')
        f.write('z, psi\n')
        for i, j in zip(z, psi): f.write('%.4f, %.4f\n' % (i, j))
        f.close()

        dtstr = str(dti).replace('.', '-')
        fname = 'mb_dt%s.csv' % dtstr
        WB = pd.DataFrame(index=t)
        WB['S'] = S
        WB['QIN_FD'] = np.nan
        WB['QOUT_FD'] = np.nan
        # These are reversed, since I am using z as depth, Celia uses elevation
        WB['QIN_BD'] = -QOUT
        WB['QOUT_BD'] = -QIN
        WB['QIN_CN'] = np.nan
        WB['QOUT_CN'] = np.nan
        WB.to_csv(fname, float_format='%12.8f')


if __name__=='__main__':
    main()