#Metropolis-Hastings
from SVDataGenerator import sv_parameter_generator as param_gen
from SVDataGenerator import sv_data_generator as data_gen

import numpy as np
import math
from scipy.stats import norm

T=500
PROPOSAL_EQUAL_CURRENT_STATE = "Proposal state and current state are equal"

parameters = param_gen()
phi = parameters[0]
sigma = parameters[1]
beta = parameters[2]
observations, real_volatility = data_gen(parameters)

def sample_proposal(t):
    tmp = np.random.randint(1, 4)
    proposal_state = None
    if tmp == 1:
        proposal_state = first_proposal(t)
    elif tmp == 2:
        proposal_state = second_proposal(t)
    elif tmp == 3:
        proposal_state = third_proposal(t)
    else:
        raise ("random out of bound")
    return proposal_state, tmp

def first_proposal(t):
    proposal = None
    if t == 0:
        var = (parameters[1] ** 2) / (1-parameters[0] ** 2)
        proposal = np.random.normal(scale=np.sqrt(var))
    else:
        v = np.random.normal()
        proposal = parameters[0] * real_volatility[t-1] + parameters[1] * v
    return proposal

def first_pdf(t, proposal):
    pdf = None
    if t>0:
        normalization = 1 / sigma*math.sqrt(2*math.pi)
        exp = proposal/sigma - phi*real_volatility[t-1]
        exp = exp**2 / -2
        pdf = normalization * math.exp(exp)
    else:
        normalization = 1 / sigma*math.sqrt(2*math.pi)
        exp = proposal**2 / -2*sigma**2
        pdf = normalization * math.exp(exp)
    return pdf

def second_proposal(t):
    proposal = None
    if t < T:
        v = np.random.normal()
        proposal = (real_volatility[t+1] - sigma * v) / phi
    else:
        proposal = real_volatility[T]
    return proposal

def second_pdf(t, proposal):
    pdf = None
    if t < T:
        normalization = phi / sigma*math.sqrt(2*math.pi)
        exp = ( real_volatility[t + 1] - phi*proposal ) / sigma
        exp = exp ** 2 / -2
        pdf = normalization * math.exp(exp)
    else:
        pdf = PROPOSAL_EQUAL_CURRENT_STATE
    return pdf


def third_proposal(t):
    proposal = None
    if t == 0:
        proposal = real_volatility[0]
    else:
        w = np.random.normal()
        proposal = 2 * np.log( abs(observations[t]) / beta * abs(w) )
    return proposal

def third_pdf(t, proposal):
    pdf = None
    if t == 0:
        pdf = PROPOSAL_EQUAL_CURRENT_STATE
    else:
        normalization = abs(observations[t]) / (beta * math.sqrt(2*math.pi))
        exp = (abs(observations[t])**2 / (2*beta**2))*math.exp(-proposal) - proposal/2
        pdf = normalization * math.exp( exp )
    return pdf

def pdf_likelihood(t, state):
    tmp = np.empty(T+1)
    for i in range(T+1):
        if i == 0:
            loc = 0
            scale = sigma**2 / (1 - phi**2)
            tmp[0] = norm.pdf(state, loc=loc, scale=scale)
        else:
            transition = norm.pdf(state, loc=phi*real_volatility[t-1], scale=sigma**2)
            likelihood = norm.pdf(state, loc=0, scale=abs(beta**2*math.exp(state)))
            tmp[i] = transition * likelihood
    return tmp


state_chain=np.empty(len(real_volatility)+1)
state_chain[0]=real_volatility[0]
for t in range(T):
    current_state=state_chain[t]
    #Sampling x'
    proposal, chosen_density=sample_proposal(t)
    # Compute acceptance probability
    stationary_proposal = pdf_likelihood(t, proposal)
    stationary_current_state = pdf_likelihood(t, current_state)
    ratio_proposal_current_st = np.prod( np.divide(stationary_proposal, stationary_current_state) )
    print(ratio_proposal_current_st)

    if chosen_density == 1:
        #Correction in denominator
        hastings_proposal = first_pdf(t, proposal)
        #Correction in numerator
        hastings_state = first_pdf(t, current_state)
    if chosen_density == 2:
        #Correction in denominator
        hastings_proposal = second_pdf(t, proposal)
        #Correction in numerator
        hastings_state = second_pdf(t, current_state)
    if chosen_density == 3:
        #Correction in denominator
        hastings_proposal = third_pdf(t, proposal)
        #Correction in numerator
        hastings_state = third_pdf(t, current_state)

    alpha=None