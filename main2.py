from SVDataGenerator import sv_parameter_generator as param_gen
from SVDataGenerator import sv_data_generator as data_gen
from ParticleFilter2 import ParticleFilter as PF

import numpy as np
np.random.seed(123)

import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

def plot_values(values, label):
    plt.plot(values[:T], "c", label=label)
    plt.plot(observations[:T], "r.", label="observations")
    plt.plot(real_volatility[:T], "b", label="real_volatility")
    plt.legend()
    plt.show()
    plt.close()

Q1=False
Q2=False
Q4=True
Q5=False


if __name__=="__main__":
    parameters=param_gen()
    observations, real_volatility=data_gen(parameters)
    # Forget beta, parameters[2]
    #parameters=[parameters[0], parameters[1], 1]

    #Instantiate particle filter
    pf = PF(parameters, observations, real_volatility)

    N = 1000
    T = 500

    #Question 1
    if Q1:
        volatility_sis, variance_sis = pf.filter(N, T, RESAMPLING=False)
        #Plot real against sis volatility
        f = plt.figure()
        plt.plot(volatility_sis, "r", label="volatility_sis")
        plt.plot(real_volatility, "b", label="real_volatility")
        plt.legend()
        plt.show()
        f.savefig("volatility_sis.pdf", bbox_inches='tight')
        plt.close()

    #Question 2
    if Q2:
        volatility_sir, variance_sir = pf.filter(N, T, RESAMPLING=True)
        # Plot real against sir volatility
        f = plt.figure()
        plt.plot(volatility_sir, "r", label="volatility_sir")
        plt.plot(real_volatility, "b", label="real_volatility")
        plt.legend()
        plt.show()
        f.savefig("volatility_sir.pdf", bbox_inches='tight')

        plt.close()
        #Plot sis and sir variance
        f = plt.figure()
        plt.plot(variance_sir, "r", label="variance_sir")
        plt.plot(variance_sis, "b", label="variance_sis")
        plt.legend()
        plt.show()
        f.savefig("volatility_comparison.pdf", bbox_inches='tight')

        plt.close()


    #Question 4
    if Q4:
        N=[100, 500, 1000]
        #T=[50, 200, 500]
        v=[]
        f = plt.figure()
        for n in N:
            vol, var = pf.filter(n, T, RESAMPLING=True)
            v.append(np.sum(var))
            plt.plot(var, label="variance_n={}".format(n))

        plt.legend()
        plt.show()
        f.savefig("volatility_n_particles.pdf", bbox_inches='tight')

        plt.close()

    if Q5:
        N=100  #Use N particles
        T=100 #Time horizon
        betas_to_evaluate = [i/10 for i in range(1, 21)]
        out=[[None]*10 for i in range(20)]
        for ctr, beta in enumerate( betas_to_evaluate ): #Choose a beta
            parameters = [parameters[0], parameters[1], beta]
            pf = PF(parameters, observations[:T], real_volatility[:T])
            for i in range(10): #Run each beta 10 times
                out[ctr][i] = pf.log_likelihood(N, T)
        f = plt.figure()
        plt.boxplot(out)
        plt.show()
        f.savefig("boxplot.pdf", bbox_inches='tight')

