ACTOR_LR=0.0005
BACL="0_0_0_0_0"
ACL="0.2_0.4_0.6_0.8_1.0"

for NOISE in 0.1
do
    for REG_W in 32 64 128
    do
        for SEED in 7 11 13 17 19
	do
	python test_dvd3pg.py\
		--env_path output/kr/env/kr_user_env_lr0.001_reg0.0003.env\
		--initial_temper 20\
		--urm_log_path output/kr/env/log/kr_user_env_lr0.001_reg0.0003.model.log\
		--save_path output/kr/agents/uoep/uoep_SASRec_actor${ACTOR_LR}_critic0.001_niter50000_reg0.00001_ep0_noise${NOISE}_bs64_epbs32_step20_reg${REG_W}_bacl${BACL}_acl${ACL}_seed${SEED}/model\
		--population_size 5\
		--env_class KREnvironment_GPU\
		--policy_class SASRec\
		--critic_class DeterministicNN_IQN\
		--agent_class UOEP\
		--reg_weight ${REG_W}\
		--below_alpha_cvar_list ${BACL}\
		--alpha_cvar_list ${ACL}\
		--facade_class OneStageFacade\
		--seed 12\
		--cvar 1.0\
		--cuda 3\
		--max_step_per_episode 20\
		--initial_temper 20\
		--reward_func mean_with_cost\
		--sasrec_n_layer 2\
		--sasrec_d_model 32\
		--sasrec_n_head 4\
		--sasrec_dropout 0.1\
		--critic_hidden_dims 256 64\
		--slate_size 9\
		--buffer_size 100000\
		--start_timestamp 2000\
		--noise_var ${NOISE}\
		--empty_start_rate 0\
		--episode_batch_size 32\
		--batch_size 64\
		--actor_lr ${ACTOR_LR}\
		--critic_lr 0.001\
		--behavior_lr 0.00001\
		--actor_decay 0.00003\
		--critic_decay 0.00003\
		--behavior_decay 0.00003\
		--target_mitigate_coef 0.01\
		--gamma 0.9\
		--n_iter 50000\
		--initial_greedy_epsilon 0\
		--final_greedy_epsilon 0\
		--elbow_greedy 0.1\
		--check_episode 10\
		--topk_rate 1\
		--test True
	done
    done
done
