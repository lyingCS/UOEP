from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from model import *
from reader import *
import os
import utils
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--model', type=str, default='RL4RSUserResponse', help='User response model class.')
    init_parser.add_argument('--reader', type=str, default='RL4RSDataReader', help='Data reader class')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    readerClass = eval('{0}.{0}'.format(initial_args.reader))

    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=128, help='number of epoch')
    #     mode_group = parser.add_mutually_exclusive_group(required=True)
    #     mode_group.add_argument("--train", action="store_true", help="Run train")
    #     mode_group.add_argument("--train_and_eval", action="store_true", help="Run train")
    #     mode_group.add_argument("--continuous_train", action="store_true", help="Run continous train")
    #     mode_group.add_argument("--eval", action="store_true", help="Run eval")

    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    args, _ = parser.parse_known_args()
    print(args)

    utils.set_random_seed(args.seed)

    reader = readerClass(args)
    print(reader.get_statistics())

    device = 'cuda:2'
    model = modelClass(args, reader, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    try:
        epo = 0
        while epo < args.epoch:
            epo += 1
            print(f"epoch {epo} training")
            # train an epoch
            model.train()
            reader.set_phase("train")
            train_loader = DataLoader(reader, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True,
                                      num_workers=reader.n_worker)
            t1 = time()
            pbar = tqdm(total=len(train_loader.dataset), ncols=50)
            step_loss = []
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                wrapped_batch = utils.wrap_batch(batch_data, device=device)
                if epo == 0 and i == 0:
                    utils.show_batch(wrapped_batch)
                out_dict = model.do_forward_and_loss(wrapped_batch)
                loss = out_dict['loss']
                loss.backward()
                step_loss.append(loss.item())
                optimizer.step()
                pbar.update(args.batch_size)
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i + 1}, loss: {np.mean(step_loss[-100:])}")
            pbar.close()
            print("Epoch {}; time {:.4f}".format(epo, time() - t1))

            # check validation and test set
            t2 = time()
            # print(f"epoch {epo} validating")
            reader.set_phase("val")
            eval_loader = DataLoader(reader, batch_size=args.batch_size,
                                     shuffle=False, pin_memory=False,
                                     num_workers=reader.n_worker)
            valid_probs, valid_true = [], []
            pbar = tqdm(total=len(eval_loader.dataset), ncols=50)
            with torch.no_grad():
                for i, batch_data in enumerate(eval_loader):
                    # predict
                    wrapped_batch = utils.wrap_batch(batch_data, device=device)
                    out_dict = model.forward(wrapped_batch)
                    valid_probs.append(out_dict['probs'].cpu().numpy())
                    valid_true.append(batch_data['feedback'].cpu().numpy())
                    pbar.update(args.batch_size)
            pbar.close()
            auc = roc_auc_score(np.concatenate(valid_true), np.concatenate(valid_probs))
            print(f"epoch {epo} validating" + "; auc: {:.4f}".format(np.mean(auc)))

            # save model
            model.save_checkpoint()

    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)
