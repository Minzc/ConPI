import json
import os
import pickle
import sys
import time
import copy
import pickle
import argparse
import numpy as np
import networkx as nx
from datetime import datetime
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from scipy.special import softmax
import loader
import utils
# import models.context_model as context_model
from context_model import ContextInteractionModel
import train_utils
import random

random.seed(2020)


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


parser = argparse.ArgumentParser(description='process user given parameters')
parser.register('type', 'bool', str2bool)
parser.add_argument("--random_seed", type=float, default=2020)
parser.add_argument("--sample_seed", type=float, default=2020)

parser.add_argument('--num_oov', type=int, default=2000)
parser.add_argument('--re_sample_test', type='bool', default=False)
parser.add_argument('--train_neg_num', type=int, default=50)
parser.add_argument('--test_neg_num', type=int, default=100)
parser.add_argument("--num_contexts", type=int, default=100, help="# contexts for interaction")
parser.add_argument('--max_contexts', type=int, default=1000, help='max contexts to look at')
parser.add_argument('--context_gamma', type=float, default=1)
# model parameters
parser.add_argument('--ngram_embed_dim', type=int, default=100)
parser.add_argument('--n_grams', type=str, default='2, 3, 4')
parser.add_argument("--word_embed_dim", type=int, default=100, help="embedding dimention for word")
parser.add_argument('--node_embed_dim', type=int, default=128)
parser.add_argument("--dropout", type=float, default=0, help="size of testing set")
parser.add_argument('--bi_out_dim', type=int, default=50, help='dim for the last bilinear layer for output')

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs for training")
parser.add_argument("--log_interval", type=int, default=2000, help='step interval for log')
parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
parser.add_argument("--early_stop_epochs", type=int, default=10)
parser.add_argument("--metric", type=str, default='map', help='mrr or map')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--min_epochs', type=int, default=30, help='minimum number of epochs')
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')

parser.add_argument('--freeze_node', type='bool', default=True)

# path to external files
# parser.add_argument('--split_graph_path', type=str, default='../data/NDFRT_DDA_train_val_test.pkl')
# parser.add_argument('--graph_path', type=str, default='../data/NDFRT_DDA_graph.pkl')
parser.add_argument("--embed_filename", type=str, default='../data/embeddings/glove.6B.100d.txt')
parser.add_argument('--node_embed_path', type=str, default='../data/NDFRT_DDA_LINE_embed.txt')
parser.add_argument('--ngram_embed_path', type=str, default='../data/embeddings/charNgram.txt')
# parser.add_argument('--restore_para_file', type=str, default='./final_pretrain_cnn_model_parameters.pkl')
# parser.add_argument('--restore_model_path', type=str, required=True, default='')
parser.add_argument('--restore_idx_data', type=str, default='')
parser.add_argument("--logging", type='bool', default=False)
parser.add_argument("--log_name", type=str, default='empty.txt')
parser.add_argument('--restore_model_epoch', type=int, default=600)
parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
parser.add_argument("--save_dir", type=str, default='./saved_models', help='save model in the best epoch or not')
parser.add_argument("--save_interval", type=int, default=5, help='intervals for saving models')

parser.add_argument('--log_reg_model', type='bool', default=False)
parser.add_argument('--max_num_ctx', type=int, default=100)
parser.add_argument('--ctx_model', type=str, default='static')

parser.add_argument('--re_sample_neg', type='bool', default=False)
parser.add_argument('--add_own_embed', type='bool', default=False)

parser.add_argument('--rela', type=str, default='clinically_associated_with')
parser.add_argument('--input_edgelist', type=str, default='')
parser.add_argument('--split_dataset_path', type=str, default='')

parser.add_argument('--graph_feature', type='bool', default=False)
parser.add_argument('--apply_sigmoid', type='bool', default=False)

parser.add_argument('--use_pair_pretraining', type='bool', default=False)
parser.add_argument('--restore_pair_embed', type=str, default='')
parser.add_argument('--ctx_embed_dim', type=int, default=128)
args = parser.parse_args()
print('args: ', args)

print('\n' + '*' * 10, 'Key parameters:', '*' * 10)
print('Use GPU? {0}'.format(torch.cuda.is_available()))
print('Process id:', os.getpid())
print('Relation: ', args.rela)
print('Embed path ', args.node_embed_path)
print('Using logistic regression? ', args.log_reg_model)
print('Using pre-trained pair embedding? ', args.use_pair_pretraining)
print('Interaction Model: ', args.ctx_model)
print('Max # contexts: ', args.max_num_ctx)
print('*' * 37)

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
args.cuda = torch.cuda.is_available()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Term to concept relation
# args.term_strings = pickle.load(open('../../SurfCon/global_mapping/term_string_mapping.pkl', 'rb'))
# args.term_concept_dict = pickle.load(open('../../SurfCon/global_mapping/term_concept_mapping.pkl', 'rb'))
# args.concept_term_dict = pickle.load(open('../../SurfCon/global_mapping/concept_term_mapping.pkl', 'rb'))

# Term id and node id
# id_mapping = open('../BioNEV/data/Clin_Term_COOC/node_list.txt').readlines()
# id_mapping = [x.strip().split() for x in id_mapping][1::]
# args.nodeX_to_termid = {int(x[0]):int(x[1]) for x in id_mapping}
# args.termid_to_nodeX = {int(x[1]):int(x[0]) for x in id_mapping}

def load_graph(if_down_sample):
    u_ids = set()
    # ================
    # Load user ids
    # ================
    with open("amazon/amzn_rate_train_feature.txt") as f:
        for ln in f:
            obj = json.loads(ln)
            u_id = int(obj["u_id"])
            u_ids.add(u_id)
    total_users = len(u_ids)
    # ========================
    # Load and remap item ids
    # =======================
    item_remap = {}
    data = []
    links = []
    max_iid = 0
    with open("amazon/amzn_rate_train_feature.txt") as f:
        for ln in f:
            obj = json.loads(ln)
            i_id = int(obj["i_id"])
            item_remap[i_id] = total_users + i_id
            data.append(
                (int(obj["u_id"]), item_remap[i_id], obj["y"])
            )
            links.append(
                (int(obj["u_id"]), item_remap[i_id])
            )
            max_iid = max(item_remap[i_id], max_iid)

    X_0 = []
    X_1 = []
    X_2 = []
    for i, j, y in data:
        if y == 1:
            X_1.append((i, j, y))
        elif y == 0:
            X_0.append((i, j, y))
        else:
            X_2.append((i, j, y))
    if if_down_sample == True:
        data = X_0 + X_1 + random.choices(X_2, k=len(X_0))
    else:
        data = X_0 + X_1 + X_2
    total = len(data)
    print(len(X_0) / total, len(X_1) / total, len(X_0) / total)
    # ========================
    # Load Knowledge graph
    # =======================
    with open("amazon/kg_final.txt") as f:
        for ln in f:
            e1, r, e2 = ln.strip().split(" ")
            e1 = int(e1)
            e2 = int(e2)
            links.append(
                (int(e1), int(e2))
            )
            item_remap[e1] = e1 + total_users
            item_remap[e2] = e2 + total_users
            max_iid = max(item_remap[e1], max_iid)
            max_iid = max(item_remap[e2], max_iid)

    i_ids = {i for i in range(max_iid) if i not in u_ids}
    print("i_ids", len(i_ids), "u_ids", len(u_ids))
    # ================
    # Construct graph
    # ===============
    args.graph = nx.Graph()
    args.graph.add_edges_from(links)
    args.all_terms = set([int(x) for x in args.graph.nodes])
    print("G.number_of_nodes()", args.graph.number_of_nodes(), "G.number_of_edges()", args.graph.number_of_edges())
    args.all_terms = u_ids | i_ids
    assert len(args.all_terms) == max_iid, (len(u_ids), len(i_ids), max_iid)
    return data, links, u_ids, i_ids


def train(model, train_data, test_data):
    print(model)
    print([name for name, p in model.named_parameters() if p.requires_grad == True])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_on_dev = 0.0

    best_test_acc, best_train_acc = 0, 0
    best_test_macro_f1, best_train_macro_f1 = 0, 0
    best_test_micro_f1, best_train_micro_f1 = 0, 0

    num_batches = len(train_data) // args.batch_size
    print('Begin trainning...')
    for epoch in range(args.num_epochs):
        # print(datetime.now().strftime("%m/%d/%Y %X"))
        model.train()
        steps = 0
        np.random.shuffle(train_data)
        train_loss = []
        train_logits = []
        train_labels = []
        for i in range(num_batches):
            train_batch = train_data[i * args.batch_size: (i + 1) * args.batch_size]
            if i == num_batches - 1:
                train_batch = train_data[i * args.batch_size::]

            t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, labels = loader.ctx_batching(train_batch)
            masks = None
            # print("labels", labels.shape, "t1_batch", t1_batch.shape, "t1_ctx_batch", len(t1_ctx_batch))

            t1_batch = torch.tensor(t1_batch, device=args.device)
            t2_batch = torch.tensor(t2_batch, device=args.device)
            labels = torch.tensor(labels).to(args.device)

            optimizer.zero_grad()
            logits, _ = model(t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, masks=masks)
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print('nan loss!')
            train_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            labels = labels.to('cpu').detach().data.numpy()

            # print(logits.shape)
            train_logits.append(logits.to('cpu').detach().numpy())
            train_labels.append(labels)

            # evaluation
            steps += 1
        train_golds = np.concatenate(train_labels)
        train_logits = np.concatenate(train_logits)
        train_preds = np.argmax(train_logits, axis=1)
        print(train_golds.shape, train_preds.shape)
        train_acc = metrics.accuracy_score(train_golds, train_preds)
        train_micro_f1 = metrics.f1_score(train_golds, train_preds, average="micro")
        train_macro_f1 = metrics.f1_score(train_golds, train_preds, average="macro")

        print("Epoch-{0}, steps-{1}: Train Loss - {2:.3}, Train F1 - {3:.4} Train Accuracy {3:.4}".
              format(epoch, steps, np.mean(train_loss), train_micro_f1 * 100, train_acc))

        dev_results = train_utils.eval_link_context(test_data, model, criterion, args)
        print("Epoch-{0}: Dev Acc {1:.4} Micro-F1 {2:.4} Macro-F1 {3:.4}".
              format(epoch, dev_results['acc'] * 100, dev_results['micro-f1'] * 100,
                     dev_results['macro-f1'] * 100), end='')

        if dev_results['micro-f1'] > best_on_dev:  # macro f1 score
            # print(datetime.now().strftime("%m/%d/%Y %X"))
            best_on_dev = dev_results['micro-f1']

            # =============
            # Test F1
            # =============
            best_test_macro_f1 = dev_results['macro-f1']
            best_test_micro_f1 = dev_results['micro-f1']

            # =============
            # Train F1
            # =============
            best_train_macro_f1 = train_macro_f1
            best_train_micro_f1 = train_micro_f1

            print("\t-- Testing: Test Acc {0:.4} Micro-F1 {1:.4} Macro-F1 {2:.4}".
                  format(dev_results['acc'] * 100, dev_results['micro-f1'] * 100,
                         dev_results['macro-f1'] * 100))

            if args.save_best:
                utils.save(model, args.save_dir, 'best', epoch)

        print("Average train loss", np.mean(train_loss), np.std(train_loss))
    return {
        "best_test_macro_f1": best_test_macro_f1,
        "best_test_micro_f1": best_test_micro_f1,
        "best_train_macro_f1": best_train_macro_f1,
        "best_train_micro_f1": best_train_micro_f1,
    }


if __name__ == '__main__':
    data, links, u_ids, i_ids = load_graph(if_down_sample=True)
    # =====================
    # Load Node Attributes
    # =====================
    with open("amazon/all_embed_matrix.p", 'rb') as embed_file:
        # 23417 * 300
        node_mat = pickle.load(embed_file)
        print("Embedding shape", node_mat.shape)
        norm = np.linalg.norm(node_mat, axis=1)
        norm = norm.reshape(-1, 1)
        norm = norm.repeat(node_mat.shape[1], axis=1)
        print("Norm", norm.shape)
        node_mat = node_mat / norm
        print("Embedding shape", node_mat.shape)
        args.pre_train_nodes = node_mat
        args.node_embed_dim = node_mat.shape[1]
        args.node_vocab_size = len(u_ids | i_ids)

    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    # accuracies = []
    # macro_f1 = []
    # micro_f1 = []

    best_train_macro_f1s, best_test_macro_f1s = [], []
    best_train_micro_f1s, best_test_micro_f1s = [], []

    for train_index, test_index in kf.split(data):
        train_pairs = [data[i] for i in train_index]
        test_pairs = [data[i] for i in test_index]

        print(len(train_pairs), len(test_pairs))

        # =====================
        # Load Node Attributes
        # =====================
        print('Begin digitalizing ...')
        # prepare digital samples
        # prepare the neighbor dict
        neighbor_dict = lambda x: args.graph.adj[x]

        # cur_dict, sample pairs
        train_data = loader.make_idx_ctx_data_zc(train_pairs, neighbor_dict, args.max_num_ctx)
        test_data = loader.make_idx_ctx_data_zc(test_pairs, neighbor_dict, args.max_num_ctx)

        print(train_data[0])

        model = ContextInteractionModel(args).to(args.device)

        rst = train(model, train_data, test_data)
        best_train_macro_f1s.append(rst["best_train_macro_f1"])
        best_train_micro_f1s.append(rst["best_train_micro_f1"])
        best_test_macro_f1s.append(rst["best_test_macro_f1"])
        best_test_micro_f1s.append(rst["best_test_micro_f1"])

    rst = {
        "best_train_macro_f1": np.mean(best_train_macro_f1s),
        "best_train_macro_f1_std": np.std(best_train_macro_f1s),
        "best_train_micro_f1": np.mean(best_train_micro_f1s),
        "best_train_micro_f1_std": np.std(best_train_micro_f1s),
        "best_test_macro_f1": np.mean(best_test_macro_f1s),
        "best_test_macro_f1_std": np.std(best_test_macro_f1s),
        "best_test_micro_f1": np.mean(best_test_micro_f1s),
        "best_test_micro_f1_std": np.std(best_test_micro_f1s),
    }
    print(rst)
