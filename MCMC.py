#Metropolis-Hastings
from SVDataGenerator2 import sv_parameter_generator as param_gen
from SVDataGenerator2 import sv_data_generator as data_gen

import numpy as np
np.random.seed(1234)
import math
from scipy.stats import norm

T=500

def sample_proposal_calc_alpha(t):
    tmp = np.random.randint(1, 4)
    proposal_state = None
    #hastings ratio is the ratio between g(x|x') and g(x'|x)
    hastings_ratio = None
    if tmp == 1:
        proposal_state = first_proposal(t)
        hastings_ratio = hastings_ratio_case_a(t, proposal_state)
    elif tmp == 2:
        proposal_state = second_proposal(t)
        hastings_ratio = hastings_ratio_case_b(t, proposal_state)
    elif tmp == 3:
        proposal_state = third_proposal(t)
        hastings_ratio = hastings_ratio_case_c(t, proposal_state)
    #Stationary ratio is the ration between p*(x') and p*(x)
    stationary_ratio = stationary_ratio_function(t, proposal_state)
    alpha = stationary_ratio * hastings_ratio
    return proposal_state, alpha

def stationary_ratio_function(t, proposal_state):
    current_state = real_volatility[t]
    ratio = None
    if t == 0:
        next_state = real_volatility[t+1]
        var = (sigma ** 2) / (1 - phi ** 2)
        p_current_state = norm.pdf(current_state, loc=0, scale=np.sqrt(var))
        p_current_to_next = norm.pdf(next_state, loc=phi*current_state, scale=sigma)
        p_proposal_state = norm.pdf(proposal_state, loc=0, scale=np.sqrt(var))
        p_proposal_to_next = norm.pdf(next_state, loc=phi*proposal_state, scale=sigma)
        ratio = (p_proposal_state * p_proposal_to_next) / (p_current_state * p_current_to_next)
    elif t == T:
        past_state = real_volatility[T-1]
        observation = observations[t]
        p_past_to_current = norm.pdf(current_state, loc=phi*past_state, scale=sigma)
        p_past_to_proposal = norm.pdf(proposal_state, loc=phi*past_state, scale=sigma)
        p_observation_given_current = norm.pdf(observation, loc=0, scale=beta * math.exp(current_state / 2) )
        p_observation_given_proposal = norm.pdf(observation, loc=0, scale=beta * math.exp(proposal_state / 2) )
        ratio = (p_past_to_proposal * p_observation_given_proposal) / (p_past_to_proposal * p_observation_given_proposal)
    else:
        next_state = real_volatility[t+1]
        past_state = real_volatility[T-1]
        observation = observations[t]
        p_past_to_current = norm.pdf(current_state, loc=phi * past_state, scale=sigma)
        p_past_to_proposal = norm.pdf(proposal_state, loc=phi * past_state, scale=sigma)
        p_observation_given_current = norm.pdf(observation, loc=0, scale=beta * math.exp(current_state / 2))
        p_observation_given_proposal = norm.pdf(observation, loc=0, scale=beta * math.exp(proposal_state / 2))
        p_current_to_next = norm.pdf(next_state, loc=phi*current_state, scale=sigma)
        p_proposal_to_next = norm.pdf(next_state, loc=phi*proposal_state, scale=sigma)
        ratio_numerator = p_past_to_proposal * p_observation_given_proposal * p_proposal_to_next
        ratio_denominator = p_past_to_current * p_observation_given_current * p_current_to_next
        ratio = ratio_numerator / ratio_denominator
    return ratio


def hastings_ratio_case_c(t, proposal_state):
    current_state = real_volatility[t]
    if t == 0:
        return 1
    else:
        p1 = math.exp(-current_state) - math.exp(-proposal_state)
        p1 *= observations[t]**2
        p1 /= 2 * beta**2
        p2 = proposal_state - current_state
        p2 /= 2
        return math.exp(p1 + p2)

def hastings_ratio_case_b(t, proposal_state):
    current_state = real_volatility[t]
    if t == T:
        return 1
    else:
        next_state = real_volatility[t+1]
        power = phi**2 * (proposal_state**2 - current_state**2) + 2*phi*next_state*(current_state - proposal_state)
        power = power / (2 * sigma**2)
        return math.exp(power)

def hastings_ratio_case_a(t, proposal_state):
    current_state = real_volatility[t]
    if t == 0:
        power = proposal_state ** 2 - current_state ** 2
        var = (sigma ** 2) / (1-phi ** 2)
        power /= 2 * var
        return math.exp(power)
    else:
        past_state = real_volatility[t - 1]
        power = proposal_state ** 2 - current_state ** 2 + 2 * phi * past_state * (current_state - proposal_state)
        power = power / (2 * sigma ** 2)
        return math.exp(power)

def first_proposal(t):
    proposal = None
    if t == 0:
        var = (sigma ** 2) / (1-phi ** 2)
        proposal = np.random.normal(scale=np.sqrt(var))
    else:
        v = np.random.normal()
        proposal = phi * real_volatility[t-1] + sigma * v
    return proposal

def second_proposal(t):
    proposal = None
    if t < T:
        v = np.random.normal()
        proposal = (real_volatility[t+1] - sigma * v) / phi
    else:
        proposal = real_volatility[T]
    return proposal

def third_proposal(t):
    proposal = None
    if t == 0:
        proposal = real_volatility[0]
    else:
        w = np.random.normal()
        proposal = 2 * np.log( abs(observations[t]) / beta * abs(w) )
    return proposal

def sampler(t):
    # Sampling x'
    proposal, alpha = sample_proposal_calc_alpha(t)
    r = min(1, alpha)
    u = np.random.uniform()
    #new = None
    if u < r:
        new = proposal
    else:
        current = real_volatility[t]
        new = current
    return new

parameters = param_gen()
phi = parameters[0]
sigma = parameters[1]
beta = parameters[2]
observations, real_volatility = data_gen(parameters)
#observations = np.insert(observations, 0, np.NAN)

for i in range(5000):
    t = np.random.randint(0, T+1)
    mcmc = sampler(t)
