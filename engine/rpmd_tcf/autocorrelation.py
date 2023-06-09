import numpy as np
import random
import matplotlib.pyplot as plt

#--------------------Constants--------------------

class Constants(object):
    """
    (Default) simulation parameters for the calculation of position autocorrelation functions.
    """
    beta = 1.0
    mass = 1.0
    dt = 0.05
    hbar = 1.05457182e-34

    n_equilibrium = 100
    n_evolution = 500
    n_samples = 1000

def omegas(N,beta):
    """
    :param N: Number of ring polymer beads.
    :return: Ring polymer normal mode frequencies as an array.
    """
    omega_N = N / beta
    omegas = np.zeros(N)
    for i in range(N):
        omegas[i] = 2 * omega_N * np.sin(i * np.pi / N)
    return omegas

#------------------RPMD matrices------------------

def rpmd_C(N):
    """
    Generates the transformation matrix to transform the positions and momenta into the normal mode representation
    :param N: Number of ring polymer beads.
    :return: Normal mode transformation matrix.
    """
    C = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            if k == 0:
                C[j][k] = np.sqrt(1 / N)
            elif k > 0 and k <= N/2 - 1:
                C[j][k] = np.sqrt(2 / N) * np.cos((2 * np.pi * j * k) / N)
            elif k == N/2:
                C[j][k] = np.sqrt(1 / N) * np.power(-1, j)
            elif k >= N/2 + 1:
                C[j][k] = np.sqrt(2 / N) * np.sin((2 * np.pi * j * k) / N)
    return C


def rpmd_E(omega, mass, dt):
    """
    Takes value for omega_k and returns the corresponding evolution matrix for the symplectic integration.
    :param omega: Ring polymer normal mode frequency.
    :param mass: Bead mass.
    :param dt: Timestep.
    :return: Evolution matrix for corresponding normal mode frequency.
    """
    E = np.zeros((2,2))
    if omega == 0:
        # Deals with the omega = 0 limit of the evolution matrix, which would be undefined using numpy functions
        E[0][0] = 1
        E[0][1] = 0
        E[1][0] = dt/mass
        E[1][1] = 1
    else:
        E[0][0] = np.cos(omega*dt)
        E[0][1] = -mass*omega*np.sin(omega*dt)
        E[1][0] = (1/(mass*omega))*np.sin(omega*dt)
        E[1][1] = np.cos(omega*dt)
    return E

#------------------Initialisation------------------

def init_p(N, mass, beta):
    """
    Initialise momenta from Gaussian distribution (used for periodic resampling of momenta from Boltzmann distribution).
    :param N: Number of ring polymer beads.
    :return: Momentum drawn from Boltzmann distribution.
    """
    sigma_p = np.sqrt((mass * N) / beta)
    p_distribution = np.random.normal(0,sigma_p)
    return p_distribution

def init_x():
    """
    Initialise positions (from an arbitrary uniform distribution).
    :return: Initial position.
    """
    x_distribution = random.randint(-10,10) / 100
    return x_distribution

#------------------Autocorrelation Function------------------

class AutoCorrelation:
    def __init__(self, force, beta=Constants.beta, dt=Constants.dt, n_samp=Constants.n_samples,
                 n_equil=Constants.n_equilibrium, n_evol=Constants.n_evolution):
        """
        Initialise simulation parameters for calculation of TCFs.
        :param force: Timestep.
        :param n_samp: Number of samples.
        :param n_equil: Number of timesteps in equilibration phase.
        :param n_evol: Number of timesteps in evolution phase.
        """
        self.mass = Constants.mass
        self.beta = beta
        self.dt = dt
        self.n_samp = n_samp
        self.n_evol = n_evol
        self.n_equil = n_equil
        self.force = force


    def classical_autocorrelation(self):
        """
        Compute <x(0)x(t)> for a classical particle in a given 1D potential.
        :return: Array of average <x(0)x(t)> values for each timestep in the evolution phase.
        """

        # Set number of beads to 1 (classical particle)
        N = 1

        # Initialise array of x(0)x(t) values, as well as momentum for classical particle
        xx = np.zeros(self.n_evol)

        # Initialise particle position
        x = 0

        # Equilibriation phase
        for i_equilibrium in range(self.n_equil):
            # Momentum resampled every cycle to avoid nonergodicity
            p = init_p(N, self.mass, self.beta)

            # Velocity verlet algorithm
            for i_evol in range(self.n_evol):
                p += -(self.dt / 2) * self.force(x)
                x += self.dt * (p / self.mass)
                p += -(self.dt / 2) * self.force(x)

        # Sampling phase
        for i_sample in range(self.n_samp):
            # Resampling momentum each cycle
            p = init_p(N, self.mass, self.beta)

            # Setting A to current x value i.e. x(0)
            A = x

            # Velocity verlet
            for i_evol in range(self.n_evol):
                p += -(self.dt / 2) * self.force(x)
                x += self.dt * (p / self.mass)
                p += -(self.dt / 2) * self.force(x)

                # Setting B to current x value i.e. x(t)
                B = x

                # Adding to x(0)x(t) array for corresponding time value
                xx[i_evol] += A * B

        return xx


    def rpmd_autocorrelation(self, N):
        """
         Compute <x(0)x(t)> for an N-bead ring polymer in a given 1D potential.
        :return: Array of average <x(0)x(t)> values for each timestep in the evolution phase.
        """

        # Create matrix objects for the trajectory propagator
        RPMD_C = rpmd_C(N)
        omega_list = omegas(N, self.beta)

        # Initialise array for the x(0)x(t) data, as well as the (p,x) vectors for each bead to be used in the normal mode basis
        xx = np.zeros(self.n_evol)
        px_vectors = np.zeros((N, 2))

        # Initialise arrays to store momentum and position of the beads
        p = np.zeros(N)
        x = np.zeros(N)

        # Initialise x
        for i in range(N):
            x[i] = 0

        # Equilibration phase
        for i_equilibrium in range(self.n_equil):

            # Resample momenta from Boltzmann distribution
            for i in range(N):
                p[i] = init_p(N, self.mass, self.beta)

            # Velocity Verlet with normal mode transformations
            for i_segment in range(self.n_evol):
                p += -(self.dt / 2) * self.force(x)

                px_vectors[:, 0] = np.dot(p, RPMD_C)
                px_vectors[:, 1] = np.dot(x, RPMD_C)

                for i in range(N):
                    px_vectors[i, :] = np.dot(rpmd_E(omega_list[i], self.mass, self.dt), px_vectors[i, :])

                p = np.dot(RPMD_C, px_vectors[:, 0])
                x = np.dot(RPMD_C, px_vectors[:, 1])

                p += -(self.dt / 2) * self.force(x)

        # Sampling phase
        for i_sample in range(self.n_samp):

            # Resample momenta from Boltzmann distribution
            for i in range(N):
                p[i] = init_p(N, self.mass, self.beta)

            A = 0
            for i in range(N):
                A += x[i]
            A /= N

            # Velocity verlet
            for i_evolution in range(self.n_evol):
                p += -(self.dt / 2) * self.force(x)

                px_vectors[:, 0] = np.dot(p, RPMD_C)
                px_vectors[:, 1] = np.dot(x, RPMD_C)

                for i in range(N):
                    px_vectors[i, :] = np.dot(rpmd_E(omega_list[i], self.mass, self.dt), px_vectors[i, :])

                p = np.dot(RPMD_C, px_vectors[:, 0])
                x = np.dot(RPMD_C, px_vectors[:, 1])

                p += -(self.dt / 2) * self.force(x)

                B = 0
                for i in range(N):
                    B += x[i]
                B /= N

                # Add x(0)x(t) value to the xx array
                xx[i_evolution] += (A * B)

        return xx

if __name__ == '__main__':
    time = np.zeros(Constants.n_evolution)
    for i_evolution in range(Constants.n_evolution):
        time[i_evolution] = Constants.dt * i_evolution

    # Define potential derivative function:
    def f(x):
        return x**3

    # Initialise ACF class and plot classical and RPMD TCFs.
    acf = AutoCorrelation(f)
    classical_xx = acf.classical_autocorrelation()
    rpmd_xx = acf.rpmd_autocorrelation(N=4)
    plt.plot(time,classical_xx/acf.n_samp,color='b',label='Classical')
    plt.plot(time,rpmd_xx/acf.n_samp,color='r',label='RPMD')
    plt.legend()
    plt.show()