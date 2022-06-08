import io
import time

import click

from models import *
from utils import *


def load_mat_dataset(dataname, datatype):
    """
    Load bundle matching datasets
    """
    path = f'../data/{dataname}/{datatype}'
    user_bundle_trn = load_obj(f'{path}/train.pkl') #
    user_bundle_vld = load_obj(f'{path}/valid.pkl') #
    user_bundle_test = load_obj(f'{path}/test.pkl') #
    user_item = load_obj(f'{path}/user_item.pkl')
    bundle_item = load_obj(f'{path}/bundle_item.pkl') #
    user_bundle_neg = np.array(load_obj(f'{path}/neg.pkl')) #
    n_user, n_item = user_item.shape
    n_bundle, _ = bundle_item.shape

    # filter
    user_bundle_vld, vld_user_idx = user_filtering(user_bundle_vld,
                                                   user_bundle_neg)
    user_bundle_test, test_user_idx = user_filtering(user_bundle_test,
                                                     user_bundle_neg)
    return n_user, n_item, n_bundle, bundle_item, user_item,\
           user_bundle_trn, user_bundle_vld, user_bundle_test,\
           vld_user_idx, test_user_idx


def load_gen_dataset(dataname, datatype, num_pos):
    """
    Load bundle generation datasets
    """
    path = f'../data/{dataname}/{datatype}'
    gen_input = load_obj(f'{path}/gen_bundle_item_input.pkl')
    gen_output = load_obj(f'{path}/gen_bundle_item_output.pkl')
    gen_neg = load_obj(f'{path}/gen_neg.pkl')
    gen_user_bundle = load_obj(f'{path}/gen_user_bundle.pkl')
    gen_pos = np.zeros((gen_neg.shape[0], num_pos))
    rows, cols = gen_output.nonzero()
    gen_pos_dict = {}
    for row, col in zip(rows, cols):
        if row not in gen_pos_dict:
            gen_pos_dict[row] = [col]
        else:
            gen_pos_dict[row].append(col)
    for key, value in gen_pos_dict.items():
        gen_pos[key] = np.array(value)
    gen_pos = gen_pos.astype(np.int32)
    gen_result = np.concatenate((gen_pos, gen_neg), axis=1)
    return gen_input, gen_result, gen_user_bundle


def user_filtering(csr, neg):
    """
    Filter out users for validation and test
    """
    idx, _ = np.nonzero(np.sum(csr, 1))
    pos = np.nonzero(csr[idx].toarray())[1]
    pos = pos[:, np.newaxis]
    neg = neg[idx]
    arr = np.concatenate((pos, neg), axis=1)
    return arr, idx


@click.command()
@click.option('--data', type=str, default='youshu')
@click.option('--task', type=str, default='mat')
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=200)
def main(data, task, seed, epochs):
    """
    Main function
    """
    set_seed(seed)
    n_user, n_item, n_bundle, bundle_item, user_item,\
    user_bundle_trn, user_bundle_vld, user_bundle_test,\
    vld_user_idx, test_user_idx = load_mat_dataset(data, task)
    ks = [5, 10, 20]
    if data == 'steam':
        num_pos = 1
    elif data == 'youshu':
        num_pos = 5
    elif data == 'netease':
        num_pos = 10

    config = {'n_item': n_item,
              'emb_dim': 200,
              'lr': 1e-3,
              'decay': 1e-5,
              'batch_size': 2000,
              'drop': 0.3,
              'num_pos': num_pos}
    model = BundleMage(**config).to(DEVICE)
    model.get_mat_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, user_bundle_test,
                          vld_user_idx, test_user_idx)

    if task == 'gen':
        gen_input, gen_result, gen_user_bundle = load_gen_dataset(data, task, num_pos)
        model.get_gen_dataset(gen_input, gen_result, gen_user_bundle)

    trn_start_time = time.time()
    header = f' Epoch   |          losses           '

    for k in ks:
        header += f'|   Recall@{k:2d}   |    NDCG@{k:2d}    '
    header += '|'
    print(header)
    saved_model, best_vld_acc, best_content, best_gen_content = io.BytesIO(), 0., '', ''

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        trn_loss = model.update_model()
        epoch_elapsed = time.time() - epoch_start_time
        trn_elapsed = time.time() - trn_start_time
        vld_loss, test_loss, vld_recalls, vld_ndcgs, test_recalls, test_ndcgs = \
            model.evaluate_mat(ks=ks)

        if task == 'gen':
            gen_recalls, gen_ndcgs = model.evaluate_gen(ks=ks)
        content = f'{epoch:3d} Rec | {trn_loss:8.4f} {vld_loss:8.4f} {test_loss:8.4f} '

        for vld_recall, vld_ndcg, test_recall, test_ndcg in \
                zip(vld_recalls, vld_ndcgs, test_recalls, test_ndcgs):
            content += f'| {vld_recall:.4f} {test_recall:.4f} | {vld_ndcg:.4f} {test_ndcg:.4f} '
        content += f'| {epoch_elapsed:10.2f} {trn_elapsed:10.2f}'

        if task == 'gen':
            gen_content = f'{epoch:3d} Gen |                            |'
            for gen_recall, gen_ndcg in \
                    zip(gen_recalls, gen_ndcgs):
                gen_content += f'        {gen_recall:.4f} | ' \
                               f'       {gen_ndcg:.4f} |'

        if vld_recalls[0] > best_vld_acc:
            best_vld_acc = vld_recalls[0]
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)
            best_content = content
            if task == 'gen':
                best_gen_content = gen_content

        if epoch % 1 == 0:
            print(content)
            if task == 'gen':
                print(gen_content)

        if epoch % 20 == 0:
            print('============================ CUR BEST ============================')
            print(best_content)
            if task == 'gen':
                print(best_gen_content)
            print('=================================================================')

    print('============================ BEST ============================')
    print(best_content)

    if task == 'gen':
        print(best_gen_content)


if __name__ == '__main__':
    main()
