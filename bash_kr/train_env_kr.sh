mkdir -p output
mkdir -p output/kr/
mkdir -p output/kr/env
mkdir -p output/kr/env/log

data_path="dataset/kr/"
output_path="output/kr/"

for REG in 0.0001 0.0003 0.001
do
    for LR in 0.0003 0.001 0.003
    do
        python train_env.py\
            --model KRUserResponse\
            --reader KRDataReader\
            --train_file ${data_path}KR_seq_data_train.meta\
            --val_file ${data_path}KR_seq_data_test.meta\
            --test_file ${data_path}KR_seq_data_test.meta\
            --item_meta_file ${data_path}iEmb.npy\
            --user_meta_file ${data_path}uEmb.npy\
            --data_separator '@'\
            --meta_data_separator ' '\
            --loss 'bce'\
            --l2_coef ${REG}\
            --lr ${LR}\
            --epoch 2\
            --seed 19\
            --model_path ${output_path}env/kr_user_env_lr${LR}_reg${REG}.model\
            --max_seq_len 50\
            --n_worker 4\
            --feature_dim 16\
            --hidden_dims 256\
            --attn_n_head 2\
            > ${output_path}env/log/kr_user_env_lr${LR}_reg${REG}.model.log
    done
done
