import numpy as np
from numpy.random import seed
seed(123)

from math import sqrt, pi

class ParticleFilter(object):

    def __init__(self,
                 parameters,
                 observations,
                 real_volatility
                ):
        self.phi = parameters[0]
        self.sigma = parameters[1]
        self.beta = parameters[2]
        self.observations=observations
        self.real_volatility=real_volatility

    def filter(self,
               N, #Number of particles to use
               T, #length of the sequence
               RESAMPLING=True,
              ):
        assert T <= len(self.observations), "T is longer than observations sequence"
        particles = np.empty(N, dtype=np.float64)
        volatility = np.empty(T, dtype=np.float64)
        weights = np.ones(N, dtype=np.float64)
        variance = np.empty(T, dtype=np.float64)
        for t in range(T):
            if t == 0:
                var = (self.sigma ** 2) / (1 - self.phi ** 2)
                particles = np.random.normal(scale=sqrt(var),
                                             size=N)
            else:
                for i in range(N):
                    particles[i] = np.random.normal(loc=self.phi * particles[i],
                                                    scale=self.sigma)
            weights = weights * self._likelihood(N, particles, self.observations[t])
            # Normalize weights array
            total = np.sum(weights)
            weights /= total
            # Resampling
            if RESAMPLING:
                if self._ess(weights) < (N / 2):
                    particles, weights, _ = self.__systematic_resample(N, particles, weights)

            # Estimate volatility
            volatility[t] = np.sum(np.multiply(particles, weights))
            # Estimate variance
            # Variance is calculated subtracting the real observation
            # not the volatility
            error = np.subtract(particles, volatility[t])
            variance[t] = np.average(error ** 2, weights=weights)
        return volatility, variance

    def log_likelihood(self,
                       N, #Number of particles to use
                       T, #length of the sequence
                      ):
        assert T <= len(self.observations), "T is longer than observations sequence"
        particles = np.empty(N, dtype=np.float64)
        tmp = np.empty(T, dtype=np.float64)
        weights = np.ones(N, dtype=np.float64)
        for t in range(T):
            if t == 0:
                var = (self.sigma ** 2) / (1 - self.phi ** 2)
                particles = np.random.normal(scale=sqrt(var),
                                             size=N)
            else:
                for i in range(N):
                    particles[i] = np.random.normal(loc=self.phi * particles[i],
                                                    scale=self.sigma)
            weights = weights * self._likelihood(N, particles, self.observations[t])

            total = np.sum(weights)
            norm_weights = np.divide(weights, total)

            if self._ess(norm_weights) < (N / 2):
                particles, _ , idx = self.__systematic_resample(N, particles, norm_weights)
                weights = weights[idx]
            tmp[t] = np.log(np.sum(weights)) - np.log(N)
        return np.sum(tmp)

    # likelihood: P(Y_t | particles)
    def _likelihood(self, N, particles, y):
        likelihood = np.ones(N)
        likelihood = np.multiply(likelihood, y ** 2)
        variance = np.multiply(np.exp(particles), self.beta ** 2)
        likelihood = np.divide(likelihood, np.multiply(-2, variance))
        likelihood = np.exp(likelihood)
        likelihood = np.divide(likelihood, np.sqrt(2 * pi * variance))
        return likelihood

    def __systematic_resample(self, N, particles, weights):
        to_be_chosen_idx = np.array( range(N) )
        #index of chosen particles
        idx = np.random.choice(to_be_chosen_idx, N, replace=True, p=weights)
        particles=particles[idx]
        weights.fill( 1 / N )
        return particles, weights, idx

    def _ess(self, weights):
        tmp = np.sum(np.power(weights, 2))
        return 1 / tmp