from parameterFitting import *
from simulation import *
import matplotlib.pyplot as plt
import seaborn as sns


fig_save = '/home/wholebrain/Desktop/modeling_figures'
files = list_files('RR_Data', 'csv')
fully_trained_m1 = []
fully_trained_m2 = []
for f in files:
    if 'epoch-5' in f and '/trials' in f:
        if 'RRM028' in f:
            fully_trained_m1.append(f)
        if 'RRM029' in f:
            fully_trained_m2.append(f)
fully_trained_m1.sort()
fully_trained_m2.sort()

mouse_1_full, m1_sessions, m1_trials = stack_data(fully_trained_m1)
mouse_2_full, m2_sessions, m2_trials = stack_data(fully_trained_m2)
R_mouse1 = restaurant_val(mouse_1_full)
R_mouse2 = restaurant_val(mouse_2_full)
mouse_1_satiated, mouse_1pelletcount = satiated_trials(mouse_1_full)
mouse_2_satiated, mouse_2pelletcount = satiated_trials(mouse_2_full)
mouse_1_committed = committed_trials(mouse_1_full)
mouse_2_committed = committed_trials(mouse_2_full)

print('== mouse_1 ==')
print('ID: RRM028')
print('sessions:', m1_sessions)
print('trials:', m1_trials)
print()
print('== mouse_2 ==')
print('ID: RRM029')
print('sessions:', m2_sessions)
print('trials:', m2_trials)


"""pick dataset you want to fit the model on"""
mouse_1 = mouse_1_committed
mouse_2 = mouse_2_committed


""""""""""""""""""""""""""
"""""Fit to Real Data"""""
""""""""""""""""""""""""""

# set up the optimizer
alphamin = 0.01
alphamax = 1.01
deltamin = 0.01
deltamax = 1.01
gammamin = -0.5
gammamax = 0.501
betamin = 0.01
betamax = 1.01
bounds_2 = [[alphamin, alphamax], [deltamin, deltamax], [gammamin, gammamax], [betamin, betamax]]
bounds_1 = [[alphamin, alphamax], [deltamin, deltamax], [betamin, betamax]]
niter = 20

"""RRM028"""
bestparameters1_RRM028, bestllh1_RRM028 = optimize(llh, restaurant_val(mouse_1), bounds_1, mouse_1, niter, toplot=True)
bestparameters2_RRM028, bestllh2_RRM028 = optimize(llh, restaurant_val(mouse_1), bounds_2, mouse_1, niter, toplot=True)

print('RRM028')
print('best loglikelihood for model 1 is {0}'.format(bestllh1_RRM028))
print('best loglikelihood for model 2 is {0}'.format(bestllh2_RRM028))
print('best alpha for model 1 is {0}, best alpha for model 2 is {1}'.format(bestparameters1_RRM028[0], bestparameters2_RRM028[0]))
print('best delta for model 1 is {0}, best delta for model 2 is {1}'.format(bestparameters1_RRM028[1], bestparameters2_RRM028[1]))
print('best beta for model 1 is {0}, best beta for model 2 is {1}'.format(bestparameters1_RRM028[-1], bestparameters2_RRM028[-1]))
print('best gamma for model 2 is {0}'.format(bestparameters2_RRM028[2]))

"""RRM029"""
bestparameters1_RRM029, bestllh1_RRM029 = optimize(llh, restaurant_val(mouse_1), bounds_1, mouse_2, niter, toplot=True)
bestparameters2_RRM029, bestllh2_RRM029 = optimize(llh, restaurant_val(mouse_1), bounds_2, mouse_2, niter, toplot=True)

print('RRM029')
print('best loglikelihood for model 1 is {0}'.format(bestllh1_RRM029))
print('best loglikelihood for model 2 is {0}'.format(bestllh2_RRM029))
print('best alpha for model 1 is {0}, best alpha for model 2 is {1}'.format(bestparameters1_RRM029[0], bestparameters2_RRM029[0]))
print('best delta for model 1 is {0}, best delta for model 2 is {1}'.format(bestparameters1_RRM029[1], bestparameters2_RRM029[1]))
print('best beta for model 1 is {0}, best beta for model 2 is {1}'.format(bestparameters1_RRM029[-1], bestparameters2_RRM029[-1]))
print('best gamma for model 2 is {0}'.format(bestparameters2_RRM029[2]))


""""""""""""""""""""""""""
"""Generate and Recover"""
""""""""""""""""""""""""""
alphas = np.arange(0.01, 1.01, 0.1)
deltas = np.arange(0.01, 1.01, 0.1)
gammas = np.arange(-0.5, 0.501, 0.1)
betas = np.arange(0.1, 1, 0.1).tolist()

alphamin = 0.01
alphamax = 0.99
deltamin = 0.01
deltamax = 0.99
gammamin = -0.49999
gammamax = 0.49999
betamin = 0
betamax = 1
niter = 20

#number of sessions per simulation
nsessions = 5
# number of trials per session
trials = 300
# number of simulations
niter = 50
# number of starting points for optimizer
nstartingpoints = 5

"""generate and recover for model 1"""

bounds = [[alphamin, alphamax], [deltamin, deltamax], [betamin, betamax]]
# results of the generate and recover procedure
genrec = np.full((niter, 6), np.nan)
optimcurve = np.full((niter, nstartingpoints), np.nan)

for i in range(niter):
    alpha = np.random.choice(alphas)
    delta = np.random.choice(deltas)
    beta = np.random.choice(betas)
    param = [alpha, delta, beta]
    Data, R = simulate(param, nsessions, trials)
    Data = pd.DataFrame(Data, columns=['session', 'trial', 'r', 'p', 'choice', 'U', 'choice_p'])

    outcomes = np.full((nstartingpoints, 4), np.nan)
    for s in range(nstartingpoints):
        params0 = [np.random.rand(), np.random.rand(), np.random.rand()]
        results = minimize(fun=llh_reparameterized, x0=params0, bounds=bounds, args=(Data, R))
        x = results.x
        bestllh = llh_reparameterized(x, Data, R)

        outcomes[s] = [bestllh, x[0], x[1], x[2]]
        optimcurve[i, s] = np.nanmin(outcomes[:, 0])

    optimcurve[i] -= optimcurve[i, -1]
    bestparameters = outcomes[s, 1:]
    genrec[i] = [alpha, delta, beta, bestparameters[0], bestparameters[1], bestparameters[2]]

fig = plt.figure()
# plot the results
pname = ['alpha', 'delta', 'beta']
for p in range(3):
    #plot generated against recovered for each parameter
    plt.subplot(1, 3, p+1)
    sns.regplot(genrec[:, p], genrec[:, 3+p])
    # plot unity
    plt.plot(bounds[p], linewidth=2)
    plt.xlabel('true parameter')
    plt.ylabel('recovered parameter')
    plt.title(pname[p])
    plt.tight_layout()
fig.savefig(fig_save + '/generate_and_recover_model1')

# look for correlations in fit parameters
fig2 = plt.figure()
plt.subplot(1, 4, 1)
sns.regplot(genrec[:, 3], genrec[:, 4])
plt.xlabel('recovered alpha')
plt.ylabel('recovered delta')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(1, 4, 2)
sns.regplot(genrec[:, 4], genrec[:, 5])
plt.xlabel('recovered delta')
plt.ylabel('recovered beta')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(1, 4, 3)
sns.regplot(genrec[:, 3], genrec[:, 5])
plt.xlabel('recovered alpha')
plt.ylabel('recovered beta')
plt.title('correlations?')
plt.tight_layout()

# # see if the number of starting points is satisfying.
plt.subplot(1, 4, 4)
mean_optimcurve = np.mean(optimcurve, axis=0)
se_optimcurve = np.std(optimcurve, axis=0) / np.sqrt(optimcurve.shape[0])
plt.errorbar(range(len(mean_optimcurve)), np.mean(optimcurve, axis=0), yerr=se_optimcurve)
plt.xlabel('random starting point number')
plt.ylabel('optimum llh')
plt.tight_layout()

fig2.savefig(fig_save + '/correlation_model1')

"""generate and recover for model 2"""

bounds = [[alphamin, alphamax], [deltamin, deltamax], [gammamin, gammamax], [betamin, betamax]]
# results of the generate and recover procedure
genrec = np.full((niter, 8), np.nan)
optimcurve = np.full((niter, nstartingpoints), np.nan)

for i in range(niter):
    alpha = np.random.choice(alphas)
    delta = np.random.choice(deltas)
    gamma = np.random.choice(gammas)
    beta = np.random.choice(betas)
    param = [alpha, delta, gamma, beta]
    Data, R = simulate(param, nsessions, trials)
    Data = pd.DataFrame(Data, columns=['session', 'trial', 'r', 'p', 'choice', 'U', 'choice_p'])

    outcomes = np.full((nstartingpoints, 5), np.nan)
    for s in range(nstartingpoints):
        params0 = [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]
        results = minimize(fun=llh_reparameterized, x0=params0, bounds=bounds, args=(Data, R))
        x = results.x
        bestllh = llh_reparameterized(x, Data, R)

        outcomes[s] = [bestllh, x[0], x[1], x[2], x[3]]
        optimcurve[i, s] = np.nanmin(outcomes[:, 0])

    optimcurve[i] -= optimcurve[i, -1]
    bestparameters = outcomes[s, 1:]
    genrec[i] = [alpha, delta, gamma, beta, bestparameters[0], bestparameters[1], bestparameters[2], bestparameters[3]]

fig1 = plt.figure()
# plot the results
pname = ['alpha', 'delta', 'gamma', 'beta']
for p in range(4):
    #plot generated against recovered for each parameter
    plt.subplot(1, 4, p+1)
    sns.regplot(genrec[:, p], genrec[:, 4+p])
    # plot unity
    plt.plot(bounds[p], linewidth=2)
    plt.xlabel('true parameter')
    plt.ylabel('recovered parameter')
    plt.title(pname[p])
    plt.tight_layout()
fig1.savefig(fig_save + '/generate_and_recover_model2')

# look for correlations in fit parameters
fig2 = plt.figure()
plt.subplot(2, 3, 1)
sns.regplot(genrec[:, 3], genrec[:, 4])
plt.xlabel('recovered alpha')
plt.ylabel('recovered delta')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(2, 3, 2)
sns.regplot(genrec[:, 3], genrec[:, 5])
plt.xlabel('recovered alpha')
plt.ylabel('recovered gamma')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(2, 3, 3)
sns.regplot(genrec[:, 3], genrec[:, 6])
plt.xlabel('recovered alpha')
plt.ylabel('recovered beta')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(2, 3, 4)
sns.regplot(genrec[:, 4], genrec[:, 5])
plt.xlabel('recovered delta')
plt.ylabel('recovered gamma')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(2, 3, 5)
sns.regplot(genrec[:, 4], genrec[:, 6])
plt.xlabel('recovered delta')
plt.ylabel('recovered beta')
plt.title('correlations?')
plt.tight_layout()

plt.subplot(2, 3, 6)
sns.regplot(genrec[:, 5], genrec[:, 6])
plt.xlabel('recovered gamma')
plt.ylabel('recovered beta')
plt.title('correlations?')
plt.tight_layout()

fig2.savefig(fig_save + '/correlation_model2')

# # see if the number of starting points is satisfying.
fig3 = plt.figure()
mean_optimcurve = np.mean(optimcurve, axis=0)
se_optimcurve = np.std(optimcurve, axis=0) / np.sqrt(optimcurve.shape[0])
plt.errorbar(range(len(mean_optimcurve)), np.mean(optimcurve, axis=0), yerr=se_optimcurve)
plt.xlabel('random starting point number')
plt.ylabel('optimum llh')
plt.tight_layout()


""""""""""""""""""""""""""
"""""Model Comparison"""""
""""""""""""""""""""""""""

#RRM028
AIC1 = -2.0 * bestllh1_RRM028 + 2.0 * 3
AIC2 = -2.0 * bestllh2_RRM028 + 2.0 * 4

print('Mouse1 Best fit model_1 AIC={0}'.format(str((AIC1))))
print('Mouse1Best fit model_2 AIC={0}'.format(str((AIC2))))

#RRM029
AIC1 = -2.0 * bestllh1_RRM029 + 2.0 * 3
AIC2 = -2.0 * bestllh2_RRM029 + 2.0 * 4

print('Mouse2 Best fit model_1 AIC={0}'.format(str((AIC1))))
print('Mouse2 Best fit model_2 AIC={0}'.format(str((AIC2))))


"""Generate Confusion Matrix"""
# # set the number of parameter samples
nsamples = 100  # start with nsamples=2 to make sure your code works, then use 100 to draw better conclusions

# # set the number of learning sessions
nsessions = 10  # also try 10, 100
trials = 300

# set the number of starting points for the optimizer
niter = 10

# for storing results
AICs = np.empty((2, 2, nsamples))
BICs = np.empty((2, 2, nsamples))

for sa in range(nsamples):
    print(sa)

    # create the data with RL simulation with random parameters
    alpha = np.arange(0.01, 1.01, 0.1)
    delta = np.arange(0.01, 1.01, 0.1)
    gamma = np.arange(-0.5, 0.501, 0.1)
    beta = np.arange(0, 1.01, 0.1)


    param_2 = [np.random.choice(alpha), np.random.choice(delta), np.random.choice(gamma), np.random.choice(beta)]
    param_1 = [np.random.choice(alpha), np.random.choice(delta), np.random.choice(beta)]
    Data1, R1 = simulate(param_1, nsessions, trials)
    Data2, R2 = simulate(param_2, nsessions, trials)
    Data1 = pd.DataFrame(Data1, columns=['session', 'trial', 'restaurant', 'tone_prob', 'accept', 'U', 'choice_p'])
    Data2 = pd.DataFrame(Data2, columns=['session', 'trial', 'restaurant', 'tone_prob', 'accept', 'U', 'choice_p'])

    # fit the data with both model 1 and model 2. compute their AICs
    # fit with model 1
    bestparameters_1, bestllh_1 = optimize(llh, R1, bounds_1, Data1, niter, toplot=False)

    # fit with model 2
    bestparameters_2, bestllh_2 = optimize(llh, R1, bounds_2, Data1, niter, toplot=False)

    AIC_1 = -2 * bestllh_1 + 2 * 2
    AIC_2 = -2 * bestllh_2 + 2 * 3
    BIC_1 = -2 * bestllh_1 + np.log(Data1.shape[0]) * 2
    BIC_2 = -2 * bestllh_2 + np.log(Data1.shape[0]) * 3
    ###

    # store the relative AIC
    AICs[0, :, sa] = [AIC_1, AIC_2] - min(AIC_1, AIC_2)  # store the relative AIC
    BICs[0, :, sa] = [BIC_1, BIC_2] - min(BIC_1, BIC_2)  # store the relative BIC

    # do the same thing for data simulated with RL2a.m

    # create the data with RL2 simulation
    # set the parameters

    # fit with RL
    bestparameters_1, bestllh_1 = optimize(llh, R2, bounds_1, Data2, niter, False)

    # fit with RL2
    bestparameters_2, bestllh_2 = optimize(llh, R2, bounds_2, Data2, niter, False)

    AIC_1 = -2 * bestllh_1 + 2 * 2
    AIC_2 = -2 * bestllh_2 + 2 * 3
    BIC_1 = -2 * bestllh_1 + np.log(Data2.shape[0]) * 2
    BIC_2 = -2 * bestllh_2 + np.log(Data2.shape[0]) * 3

    ###

    # store the relative AIC
    AICs[1, :, sa] = [AIC_1, AIC_2] - min(AIC_1, AIC_2)  # store the relative AIC
    BICs[1, :, sa] = [BIC_1, BIC_2] - min(BIC_1, BIC_2)  # store the relative BIC


# visualize the results

fig,axes = plt.subplots(1,2,figsize=(12,6))

## AIC

fig = plt.figure()
plt.sca(axes[0])
sns.heatmap(np.mean(AICs>0,2),annot=True)
# plt.imshow(1-np.mean(AICs>0,2), cmap = plt.cm.viridis)
# plt.colorbar()
plt.xticks([0.5,1.5],['model 1','model 2'])
plt.yticks([0.5,1.5],['model 1','model 2'])
plt.xlabel('recovering model')
plt.ylabel('simulating model')
plt.title('Proportion of samples with best AIC')
fig.savefig(fig_save+'/AIC1')

## BIC

fig = plt.figure()
plt.sca(axes[1])
sns.heatmap(np.mean(BICs>0,2),annot=True)
# plt.imshow(1-np.mean(BICs>0,2), cmap = plt.cm.viridis)
# plt.colorbar()
plt.xticks([0.5,1.5],['model 1','model 2'])
plt.yticks([0.5,1.5],['model 1','model 2'])
plt.xlabel('recovering model')
plt.ylabel('simulating model')
plt.title('Proportion of samples with best BIC')

plt.tight_layout()
fig.savefig(fig_save+'/BIC1')

""""""""""""""""""""""""""
"""""Model Validation"""""
""""""""""""""""""""""""""
"""Visualize data generated by fitted parameters for RRM028"""
fig1, simulated_028_model1, placeholder  = plot_sim(bestparameters1_RRM028, 3, 300,
                                                    R=restaurant_val(mouse_1), fig_save=fig_save+'028fitted1')
fig2, simulated_028_model2, placeholder = plot_sim(bestparameters2_RRM028, 3, 300,
                                                   R=restaurant_val(mouse_1), fig_save=fig_save+'028fitted2')
ax1 = plot_pellets(simulated_028_model1, 'RRM028 Model 1 Simulated', fig_save=fig_save+'028fitted1_pellets')
ax2 = plot_prob(simulated_028_model2, 'RRM028 Model 2 Simulated',
                restaurant_val(mouse_1_committed), fig_save=fig_save+'028fitted1_probs')

"""Visualize data generated by fitted parameters for RRM029"""
fig3, simulated_029_model1, placeholder = plot_sim(bestparameters1_RRM029, 3, 300,
                                                   R=restaurant_val(mouse_2), fig_save=fig_save+'029fitted1')
fig4, simulated_029_model2, placeholder = plot_sim(bestparameters2_RRM029, 3, 300,
                R=restaurant_val(mouse_2), fig_save=fig_save+'029fitted2')
ax3 = plot_pellets(simulated_029_model1, 'RRM028 Model 1 Simulated', fig_save=fig_save+'029fitted1_pellets')
ax4 = plot_prob(simulated_029_model2, 'RRM029 Model 2 Simulated',
                restaurant_val(mouse_2_committed), fig_save=fig_save+'029fitted1_probs')

