# Reinforcing Long-Term Performance in Recommender Systems with User-Oriented Exploration Policy (SIGIR 2024)

## 0. Setup

```
conda create -n uoep python=3.9
conda activate uoep
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn
pip install tqdm
conda install -c anaconda ipykernel
python -m ipykernel install --user --name uoep --display-name "UOEP"
```


## 1. Pretrain User Response Model as Environment Component

Modify train_env.sh:
* Change the directories, data_path, and output_path for your dataset
* Set the following arguments with X in {RL4RS, ML1M}:
  * --model {X}UserResponse\
  * --reader {X}DataReader\
  * --train_file ${data_path}{X}_b_train.csv\
  * --val_file ${data_path}{X}_b_test.csv\
* Set your model_path and log_path in the script.

Run:
> bash bash_kr/train_env_kr.sh

## 2. Training

> bash bash_kr/train_uoep_kr.sh

## 3. Test

> bash bash_kr/test_uoep_kr.sh

## 4. Show Result

> python show_result.py

## Citation

Please cite our paper if you use this repository.

```
@inproceedings{zhang2024reinforcing,
  title={Reinforcing Long-Term Performance in Recommender Systems with User-Oriented Exploration Policy},
  author={Zhang, Changshuo and Chen, Sirui and Zhang, Xiao and Dai, Sunhao and Yu, Weijie and Xu, Jun},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1850--1860},
  year={2024}
}
```
