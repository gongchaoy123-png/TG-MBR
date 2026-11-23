"""
CLRN-style multi-behavior BPR training / evaluation entry point.

- Multi-behavior data loading (trn_<behavior> scipy sparse pickles, target last)
- Per-epoch re-sampling of per-behavior positive/negative items (Eq.13 triples)
- Training with multi-behavior BPR and representation L2 regularization
- Evaluation: leave-one-out protocol, 1 held-out target item + 99 sampled
  negatives per user and HR/NDCG@10 and @20

Run (Tmall, paper hyperparameters):
    python main_clrn.py --dataset Tmall --hidden_dim 32 --batch 8192 \
        --gnn_layer 3 --interaction_filter --interaction_filter_strength 0.05
"""

import datetime
import copy
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as dataloader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *a, **k: x

from Params import args
from Timelogger import log
from evaluation_data import build_test_candidates
import graph_utils
from model import CLRN

BEHAVIORS = {
    'Tmall': ['pv', 'fav', 'cart', 'buy'],
    'Taobao': ['pv', 'cart', 'buy'],
    'Beibei': ['pv', 'cart', 'buy'],
    'IJCAI': ['click', 'fav', 'cart', 'buy'],
    'IJCAI_15': ['click', 'fav', 'cart', 'buy'],
    'retailrocket': ['view', 'cart', 'buy'],
    'retail': ['pv', 'fav', 'cart', 'buy'],
    'Yelp': ['neg', 'neutral', 'tip', 'pos'],
    'yelp_tiny': ['neg', 'neutral', 'tip', 'pos'],
    'MultiInt-ML10M': ['neg', 'neutral', 'pos'],
    # retail_small has no trn_pv file in the current checkout; keep its
    # original three-behavior definition until a matching pv matrix exists.
    'retail_small': ['fav', 'cart', 'buy'],
}

device = torch.device('cuda' if (args.device == 'gpu' and torch.cuda.is_available()) else 'cpu')


class MultiBehDataset(dataloader.Dataset):
    """Sample one positive and one unobserved negative per user and behavior."""

    def __init__(self, train_pairs, behaviors_data, item_num):
        self.train_pairs = train_pairs                      # [N_buy, 2]
        self.behavior_num = len(behaviors_data)
        self.item_num = item_num
        self.num_user = behaviors_data[0].shape[0]

        self.pos_lists, self.pos_sets = [], []
        for mat in behaviors_data:
            bin_mat = mat.tocsr().copy()
            bin_mat.data[:] = 1
            indptr, indices = bin_mat.indptr, bin_mat.indices
            self.pos_lists.append([indices[indptr[u]:indptr[u + 1]] for u in range(self.num_user)])
            self.pos_sets.append([set(l.tolist()) for l in self.pos_lists[-1]])

        self.resample()

    def resample(self):
        K, U = self.behavior_num, self.num_user
        self.pos_epoch = np.full((K, U), -1, dtype=np.int64)
        self.neg_epoch = np.full((K, U), -1, dtype=np.int64)
        for b in range(K):
            for u in range(U):
                items = self.pos_lists[b][u]
                if len(items) == 0:
                    continue
                self.pos_epoch[b, u] = items[np.random.randint(len(items))]

                for _ in range(100):
                    j = np.random.randint(self.item_num)
                    if j not in self.pos_sets[b][u]:
                        self.neg_epoch[b, u] = j
                        break

                if self.neg_epoch[b, u] < 0:
                    # This only occurs for nearly dense users. Avoid passing an
                    # invalid index to the model; the user's behavior is then
                    # skipped by the positive mask in __getitem__/loss.
                    self.pos_epoch[b, u] = -1

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        user = self.train_pairs[idx, 0]
        return user, self.pos_epoch[:, user], self.neg_epoch[:, user]


def load_data():
    behaviors = BEHAVIORS[args.dataset]
    # The repository stores IJCAI under ``datasets/IJCAI`` while older
    # commands and the behavior table use ``IJCAI_15``.
    dataset_dir = {'IJCAI_15': 'IJCAI'}.get(args.dataset, args.dataset)
    trn_file = f'{args.path}{dataset_dir}/trn_'
    behaviors_data = []
    user_num = item_num = -1
    for beh in behaviors:
        with open(f'{trn_file}{beh}', 'rb') as fs:
            data = pickle.load(fs).tocsr()
        behaviors_data.append(data)
        user_num = max(user_num, data.shape[0])
        item_num = max(item_num, data.shape[1])
    with open(f'{args.path}{dataset_dir}/tst_int', 'rb') as fs:
        tst = pickle.load(fs)
    test_user = np.array([idx for idx, i in enumerate(tst) if i is not None], dtype=np.int64)
    test_item = np.array([i for i in tst if i is not None], dtype=np.int64)
    return behaviors, behaviors_data, user_num, item_num, test_user, test_item


def make_random_validation(behaviors_data, seed):
    """Hold out one random buy interaction per eligible user.

    The held-out pair is removed from every training behavior graph to avoid
    leaking the validation item through an auxiliary edge.  This is a random
    holdout, not a chronological split, because the stored sparse matrices do
    not retain timestamps.
    """
    rng = np.random.default_rng(seed)
    target = behaviors_data[-1].tocsr()
    val_pairs = []
    for u in range(target.shape[0]):
        items = target.indices[target.indptr[u]:target.indptr[u + 1]]
        if len(items) >= 2:
            val_pairs.append((u, int(items[rng.integers(len(items))])))
    val_pairs = np.asarray(val_pairs, dtype=np.int64).reshape(-1, 2)
    val_set = set(map(tuple, val_pairs.tolist()))

    filtered = []
    for mat in behaviors_data:
        work = mat.tolil(copy=True)
        for u, i in val_set:
            work[u, i] = 0
        work = work.tocsr()
        work.eliminate_zeros()
        filtered.append(work)

    train_target = filtered[-1].tocsr()
    train_u, train_i = train_target.nonzero()
    train_pairs = np.stack([train_u, train_i], axis=1)
    return filtered, train_pairs, val_pairs


@torch.no_grad()
def evaluate(model, test_user, test_item, cand):
    model.eval()
    # Retail has 147k users. Score candidates in chunks instead of creating
    # all ~14.8M user-item triples at once in the interaction MLP.
    chunk = max(int(args.eval_batch), 1)
    top_ks = (10, 20)
    max_k = max(top_ks)
    compare_base = args.interaction_filter and args.interaction_filter_diagnostics
    score_types = ('full', 'base') if compare_base else ('full',)
    hit_sums = {name: {k: 0.0 for k in top_ks} for name in score_types}
    ndcg_sums = {name: {k: 0.0 for k in top_ks} for name in score_types}
    correction_abs_sum = 0.0
    base_abs_sum = 0.0
    total = 0
    for start in range(0, test_user.shape[0], chunk):
        end = min(start + chunk, test_user.shape[0])
        user_chunk = test_user[start:end]
        cand_chunk = cand[start:end]
        users = user_chunk.unsqueeze(1).expand_as(cand_chunk).reshape(-1)
        items = cand_chunk.reshape(-1)
        scores = model.predict_target_scores(users, items, return_base=compare_base)
        if compare_base:
            full, base = scores
            correction_abs_sum += (full - base).abs().sum().item()
            base_abs_sum += base.abs().sum().item()
        else:
            scores = (scores,)
        for name, score in zip(score_types, scores):
            score = score.reshape(cand_chunk.shape)
            _, topk = score.topk(min(max_k, score.shape[1]), dim=1)
            hits = topk == 0  # The test positive occupies column 0.
            for k in top_ks:
                hits_k = hits[:, :k]
                hit_k = hits_k.any(dim=1).float()
                rank_k = hits_k.float().argmax(dim=1).float()
                hit_sums[name][k] += hit_k.sum().item()
                ndcg_sums[name][k] += (hit_k / torch.log2(rank_k + 2.0)).sum().item()
        total += end - start
    model.train()
    denominator = max(total, 1)
    metrics = {name: {} for name in score_types}
    for name in score_types:
        for k in top_ks:
            metrics[name][f'HR@{k}'] = hit_sums[name][k] / denominator
            metrics[name][f'NDCG@{k}'] = ndcg_sums[name][k] / denominator
    if compare_base:
        count = max(cand.numel(), 1)
        print(f'IF eval epoch {model.current_epoch}: '
              f'{format_metrics(metrics["base"], "base_")}  | '
              f'{format_metrics(metrics["full"], "full_")}  | '
              f'mean_abs_base {base_abs_sum / count:.5g}  '
              f'mean_abs_correction {correction_abs_sum / count:.5g}')
    return metrics['full']


def format_metrics(metrics, prefix=''):
    names = ('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20')
    return '  '.join(f'{prefix}{name} {metrics[name]:.4f}' for name in names)


def main(benchmark=None):
    if args.save_best and args.eval_protocol != 'test_early_stop' and benchmark is None:
        raise ValueError('--save_best currently requires --eval_protocol test_early_stop')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    behaviors, behaviors_data, user_num, item_num, test_user, test_item = load_data()
    if args.eval_protocol in ('fixed', 'test_early_stop'):
        # Match the previous CLRN version: use every interaction in the
        # stored training matrices.  No validation edge is removed.
        train_target = behaviors_data[-1].tocsr()
        train_u, train_i = train_target.nonzero()
        train_pairs = np.stack([train_u, train_i], axis=1)
        valid_pairs = np.empty((0, 2), dtype=np.int64)
        print(f'{args.eval_protocol} evaluation protocol: '
              'use the complete stored training set')
    else:
        behaviors_data, train_pairs, valid_pairs = make_random_validation(
            behaviors_data, args.eval_sample_seed)
    valid_user = valid_pairs[:, 0]
    valid_item = valid_pairs[:, 1]
    print(f'random validation pairs: {len(valid_pairs)} '
          f'(users={len(np.unique(valid_user))})')
    print(f'dataset: {args.dataset}  behaviors: {behaviors}  (target: {behaviors[-1]})')
    print(f'user_num: {user_num}  item_num: {item_num}  device: {device}')

    # binary matrices for degrees / sampling; normalized adjacency for propagation (Eq.1)
    bin_mats = []
    for mat in behaviors_data:
        b = mat.tocsr().copy()
        b.data[:] = 1
        bin_mats.append(b)
    target_mat = bin_mats[-1]

    print('Start building behavior matrices:', datetime.datetime.now())
    adj_list = [graph_utils.get_use(b).to(device) for b in bin_mats]
    print('End building behavior matrices:', datetime.datetime.now())

    train_dataset = MultiBehDataset(train_pairs, behaviors_data, item_num)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch,
                                         shuffle=True, num_workers=0, pin_memory=True)

    valid_cand = build_test_candidates(valid_user, valid_item, target_mat, item_num,
                                       args.eval_sample_seed)
    test_cand = build_test_candidates(test_user, test_item, target_mat, item_num,
                                      args.eval_sample_seed + 1)
    valid_user_t = torch.as_tensor(valid_user, device=device)
    valid_cand_t = torch.as_tensor(valid_cand, device=device)
    test_user_t = torch.as_tensor(test_user, device=device)
    test_cand_t = torch.as_tensor(test_cand, device=device)

    model = CLRN(user_num, item_num, adj_list).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_metadata = None
    if args.save_best and benchmark is None:
        from case_checkpoint import data_fingerprints
        checkpoint_metadata = {
            'dataset_dir': str((Path(args.path) / args.dataset).resolve()),
            'data_sha256': data_fingerprints(Path(args.path) / args.dataset, behaviors),
            'behaviors': behaviors, 'num_user': user_num, 'num_item': item_num,
            'test_users': torch.as_tensor(test_user),
            'test_candidates': torch.as_tensor(test_cand),
        }

    log('Model Prepared')
    best_metrics, best_epoch, wait = None, -1, 0
    best_state = copy.deepcopy(model.state_dict()) if benchmark is None else None
    history = {'loss': []}
    for split in ('valid', 'test'):
        for metric in ('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'):
            history[f'{split}_{metric}'] = []

    # Match the fixed-epoch protocol used by SCKGD by default.  The previous
    # implementation trained epochs 0..args.epoch and selected a validation
    # checkpoint; ``--eval_protocol valid`` keeps that alternative available.
    for epoch in range(1, args.epoch + 1):
        model.current_epoch = epoch if benchmark is None else benchmark.model_epoch(epoch)
        log(f'***************** Start epoch: {epoch} ************************')
        if benchmark is not None:
            benchmark.start_training_epoch()
        train_dataset.resample()
        epoch_loss = 0.0
        comp = {'bpr_multi': 0.0, 'reg': 0.0}
        grad_norm_max, clipped_batches = 0.0, 0
        nb = 0
        batches = tqdm(train_loader) if benchmark is None else train_loader
        for users, pos, neg in batches:
            users = users.long().to(device)
            pos = pos.long().to(device)
            neg = neg.long().to(device)

            loss, bpr_multi, reg = model.cal_loss(users, pos, neg)
            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {nb + 1}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
            except RuntimeError as error:
                raise RuntimeError(f'Gradient clipping failed at epoch {epoch}, batch {nb + 1}') from error
            if args.interaction_filter_diagnostics:
                norm = grad_norm.item()
                grad_norm_max = max(grad_norm_max, norm)
                clipped_batches += int(norm > 20)
            opt.step()
            epoch_loss += batch_loss
            comp['bpr_multi'] += bpr_multi.item()
            comp['reg'] += reg.item()
            nb += 1

        if benchmark is not None:
            benchmark.finish_training_epoch(epoch, nb)
            continue

        if args.interaction_filter_diagnostics:
            print(f'IF train epoch {epoch}: max_preclip_grad_norm {grad_norm_max:.5g}, '
                  f'clipped_batches {clipped_batches}/{nb}')
        history['loss'].append(epoch_loss)
        if args.eval_protocol == 'fixed':
            # This is monitoring only: no checkpoint is selected from the
            # test set.  The model continues to the pre-set final epoch.
            metrics = evaluate(model, test_user_t, test_item, test_cand_t)
            for name, value in metrics.items():
                history[f'test_{name}'].append(value)
            print(f'epoch {epoch}: loss {epoch_loss:.4f}  '
                  f'{format_metrics(metrics, "test_")}  | '
                  f'bpr_multi {comp["bpr_multi"]/nb:.4f}  '
                  f'reg {comp["reg"]/nb:.4f}')
            save_history(history)
            continue

        if args.eval_protocol == 'test_early_stop':
            # Reproduce the old no-validation protocol: the test set is
            # evaluated after every epoch and also controls early stopping.
            metrics = evaluate(model, test_user_t, test_item, test_cand_t)
            for name, value in metrics.items():
                history[f'test_{name}'].append(value)
            print(f'epoch {epoch}: loss {epoch_loss:.4f}  '
                  f'{format_metrics(metrics)}  | '
                  f'bpr_multi {comp["bpr_multi"]/nb:.4f}  '
                  f'reg {comp["reg"]/nb:.4f}')
            if metrics['HR@10'] > (best_metrics or {}).get('HR@10', -1.0):
                best_metrics, best_epoch, wait = metrics.copy(), epoch, 0
                if args.save_best:
                    from case_checkpoint import save_checkpoint
                    save_checkpoint(args.save_best, model, args, metrics, checkpoint_metadata)
                save_history(history)
            else:
                wait += 1
                if wait >= args.patience:
                    print(f'Early stop. best epoch {best_epoch}: '
                          f'{format_metrics(best_metrics)}')
                    break
            continue

        metrics = evaluate(model, valid_user_t, valid_item, valid_cand_t)
        for name, value in metrics.items():
            history[f'valid_{name}'].append(value)
        print(f'epoch {epoch}: loss {epoch_loss:.4f}  '
              f'{format_metrics(metrics, "valid_")}  | '
              f'bpr_multi {comp["bpr_multi"]/nb:.4f}  '
              f'reg {comp["reg"]/nb:.4f}')

        if metrics['HR@10'] > (best_metrics or {}).get('HR@10', -1.0):
            best_metrics, best_epoch, wait = metrics.copy(), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            save_history(history)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Early stop. best epoch {best_epoch}: '
                      f'{format_metrics(best_metrics)}')
                break

    if benchmark is not None:
        del opt
        model.zero_grad(set_to_none=True)
        benchmark.measure_evaluation(model, evaluate, test_user_t, test_item, test_cand_t)
        return

    if args.eval_protocol == 'test_early_stop':
        print(f'BEST TEST epoch {best_epoch}: {format_metrics(best_metrics)}')
        return

    if args.eval_protocol == 'valid':
        model.load_state_dict(best_state)
        if best_epoch > 0:
            model.current_epoch = best_epoch
    test_metrics = evaluate(model, test_user_t, test_item, test_cand_t)
    if args.eval_protocol == 'valid':
        print(f'BEST VALID epoch {best_epoch}: {format_metrics(best_metrics)}')
    else:
        print(f'FINAL FIXED-EPOCH epoch {args.epoch}: '
              f'{format_metrics(test_metrics)}')
    print(f'FINAL TEST: {format_metrics(test_metrics)}')


def save_history(history):
    save_dir = f'./History/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    name = f'CLRN-paper-{args.dataset}-lr-{args.lr}-L-{args.gnn_layer}-d-{args.hidden_dim}'
    with open(os.path.join(save_dir, name + '.his'), 'wb') as fs:
        pickle.dump(history, fs)


if __name__ == '__main__':
    main()
