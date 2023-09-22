import numpy as np
import numpy

path='./output/kr/agents/'

rwd_cnts, step_cnts = [], []
rwd_means, step_means = [], []
rwd_n_means, step_n_means = [], []
rwd2_n_means, step2_n_means = [], []
rwd4_n_means, step4_n_means = [], []
rwd5_n_means, step5_n_means = [], []
rwd_ginis, step_ginis = [], []
actor_lr='0.0005'
be_lr='0.0005'

def gini(x):
    n = len(x)
    sorted_x = np.sort(x)
    index = np.arange(1, n + 1)

    # Calculate the Gini coefficient using vectorized operations
    gini_coefficient = (np.sum((2 * index - n - 1) * sorted_x)) / (n * np.sum(sorted_x))

    return gini_coefficient

for i in ['32', '64']: 
    rwd_cnt, step_cnt = [], []
    rwd_mean, step_mean = [], []
    rwd_n_mean, step_n_mean = [], []
    rwd2_n_mean, step2_n_mean = [], []
    rwd4_n_mean, step4_n_mean = [], []
    rwd5_n_mean, step5_n_mean = [], []
    rwd_gini, step_gini = [], []
    for j in ['7','11','13','17','19']:
        folder_name='uoep/uoep_SASRec_actor'+actor_lr+'_critic0.001_niter50000_reg0.00001_ep0_noise0.1_bs64_epbs32_step20_reg'+i+'_bacl0_0_0_0_0_acl0.2_0.4_0.6_0.8_1.0_seed'+j
        a=np.load(path+folder_name+'/model.test_rewards.npy')
        b=np.load(path+folder_name+'/model.test_steps.npy')
        
        n = len(a)
        k = int(n * 0.3)
        k2 = int(n*0.2)
        k4 = int(n*0.4)
        k5 = int(n*0.5)
        rwd_result = np.mean(sorted(a)[:k])
        rwd2_result = np.mean(sorted(a)[:k2])
        rwd4_result = np.mean(sorted(a)[:k4])
        rwd5_result = np.mean(sorted(a)[:k5])
        step_result = np.mean(sorted(b)[:k])
        step2_result = np.mean(sorted(b)[:k2])
        step4_result = np.mean(sorted(b)[:k4])
        step5_result = np.mean(sorted(b)[:k5])
        rwd_cnt.append(np.sum(a+0.2<1e-5))
        step_cnt.append(np.sum(b-2<1e-5))
        rwd_mean.append(np.mean(a))
        step_mean.append(np.mean(b))
        rwd_gini.append(gini(a))
        rwd_gini.append(gini(b))
        rwd_n_mean.append(rwd_result)
        rwd2_n_mean.append(rwd2_result)
        rwd4_n_mean.append(rwd4_result)
        rwd5_n_mean.append(rwd5_result)
        step_n_mean.append(step_result)
        step2_n_mean.append(step2_result)
        step4_n_mean.append(step4_result)
        step5_n_mean.append(step5_result)

    rwd_cnts.append(np.mean(rwd_cnt))
    step_cnts.append(np.mean(step_cnt))
    rwd_means.append(np.mean(rwd_mean))
    step_means.append(np.mean(step_mean))
    rwd_ginis.append(np.mean(rwd_gini))
    step_ginis.append(np.mean(rwd_gini))
    rwd_n_means.append(np.mean(rwd_n_mean))
    step_n_means.append(np.mean(step_n_mean))
    rwd2_n_means.append(np.mean(rwd2_n_mean))
    step2_n_means.append(np.mean(step2_n_mean))
    rwd4_n_means.append(np.mean(rwd4_n_mean))
    step4_n_means.append(np.mean(step4_n_mean))
    rwd5_n_means.append(np.mean(rwd5_n_mean))
    step5_n_means.append(np.mean(step5_n_mean))

    rwd_cnts.append(np.std(rwd_cnt))
    step_cnts.append(np.std(step_cnt))
    rwd_means.append(np.std(rwd_mean))
    step_means.append(np.std(step_mean))
    rwd_ginis.append(np.std(rwd_gini))
    step_ginis.append(np.std(rwd_gini))
    rwd_n_means.append(np.std(rwd_n_mean))
    step_n_means.append(np.std(step_n_mean))
    rwd2_n_means.append(np.std(rwd2_n_mean))
    step2_n_means.append(np.std(step2_n_mean))
    rwd4_n_means.append(np.std(rwd4_n_mean))
    step4_n_means.append(np.std(step4_n_mean))
    rwd5_n_means.append(np.std(rwd5_n_mean))
    step5_n_means.append(np.std(step5_n_mean))

rwd_means = [float('{:.5f}'.format(i)) for i in rwd_means]
step_means = [float('{:.5f}'.format(i)) for i in step_means]
rwd_ginis = [float('{:.5f}'.format(i)) for i in rwd_ginis]
step_ginis = [float('{:.5f}'.format(i)) for i in step_ginis]
rwd_n_means = [float('{:.5f}'.format(i)) for i in rwd_n_means]
step_n_means = [float('{:.5f}'.format(i)) for i in step_n_means]
rwd2_n_means = [float('{:.5f}'.format(i)) for i in rwd2_n_means]
step2_n_means = [float('{:.5f}'.format(i)) for i in step2_n_means]
rwd4_n_means = [float('{:.5f}'.format(i)) for i in rwd4_n_means]
step4_n_means = [float('{:.5f}'.format(i)) for i in step4_n_means]
rwd5_n_means = [float('{:.5f}'.format(i)) for i in rwd5_n_means]
step5_n_means = [float('{:.5f}'.format(i)) for i in step5_n_means]
print(rwd_cnts)
print(step_cnts)
print(rwd_means)
print(step_means)
print(rwd2_n_means)
print(step2_n_means)
print(rwd_n_means)
print(step_n_means)
print(rwd4_n_means)
print(step4_n_means)
print(rwd5_n_means)
print(step5_n_means)
print(rwd_ginis)
print(step_ginis)

