mkdir -p output

# KR environment

mkdir -p output/kr/
mkdir -p output/kr/env/
mkdir -p output/kr/env/log/
mkdir -p output/kr/agents/
mkdir -p output/kr/agents/uoep/

output_path="output/kr/"
log_name="kr_user_env_lr0.001_reg0.0003"


N_ITER=50000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0
REG=0.00003
NOISE=0.01
ELBOW=0.1
EP_BS=32
BS=64
SEED=17
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
#BEHAVE_LR=0.00001
BEHAVE_LR=0.0005
TEMPER_SWEET_POINT=0.9
REG_W=2
BACL="0_0_0_0_0"
ACL="0.2_0.4_0.6_0.8_1.0"

for NOISE in 0.1
do
    for REG in 0.00001
    do
        for INITEP in 0
        do
            for CRITIC_LR in 0.001
            do
                for ACTOR_LR in 0.0005
                do
                    for REG_W in 32 64 128
                    do
                        for SEED in 11 13 17 19 7
                        do
                        mkdir -p ${output_path}agents/uoep/uoep_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_reg${REG_W}_bacl${BACL}_acl${ACL}_seed${SEED}/

                        python train_uoep.py\
                            --env_class KREnvironment_GPU\
                            --policy_class ${SCORER}\
                            --critic_class DeterministicNN_IQN\
                            --agent_class UOEP\
                            --facade_class OneStageFacade\
			    --population_size 5\
			    --reg_weight ${REG_W}\
			    --below_alpha_cvar_list ${BACL}\
			    --alpha_cvar_list ${ACL}\
                            --seed ${SEED}\
                            --cuda 3\
                            --env_path ${output_path}env/${log_name}.env\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --reward_func mean_with_cost\
                            --urm_log_path ${output_path}env/log/${log_name}.model.log\
                            --sasrec_n_layer 2\
                            --sasrec_d_model 32\
                            --sasrec_n_head 4\
                            --sasrec_dropout 0.1\
                            --critic_hidden_dims 256 64\
                            --slate_size 9\
                            --buffer_size 100000\
                            --start_timestamp 2000\
                            --noise_var ${NOISE}\
                            --empty_start_rate ${EMPTY}\
                            --save_path ${output_path}agents/uoep/uoep_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_reg${REG_W}_bacl${BACL}_acl${ACL}_seed${SEED}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --behavior_lr ${BEHAVE_LR}\
                            --actor_decay ${REG}\
                            --critic_decay ${REG}\
                            --behavior_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --topk_rate ${TOPK}\
                            > ${output_path}agents/uoep/uoep_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_reg${REG_W}_bacl${BACL}_acl${ACL}_seed${SEED}/log
			done
                    done
                done
            done
        done
    done
done
