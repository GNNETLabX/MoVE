import torch
import numpy as np
import pandas as pd
from models.utils.mutils import is_empty_edges
from models.utils.inits import prepare


def evaluate(args, runner, num_runs):
    all_train_auc, all_val_auc, all_test_auc = [], [], []
    all_train_acc, all_val_acc, all_test_acc = [], [], []

    for runs in range(num_runs):
        data = runner.data["test"]

        filepath = "../checkpoint/" + 'MoVE' + args.dataset + str(runs) + ".pth"
        checkpoint = torch.load(filepath)
        runner.model.load_state_dict(checkpoint["model_state_dict"])
        runner.model.eval()

        train_auc_list, val_auc_list, test_auc_list = [], [], []
        train_acc_list, val_acc_list, test_acc_list = [], [], []

        for i in range(runner.len):
            embeddings, _, _ = runner.model.encode(runner.x[i], data['edge_index_list'][i].long().to(args.device))

            if i < runner.len - 1:
                z = embeddings
                edge_index, pos_edge, neg_edge = prepare(data, i + 1)[:3]
                if is_empty_edges(neg_edge):
                    continue
                auc, ap = runner.loss.predict(z, pos_edge, neg_edge, runner.model.encoder.edge_decoder)

                if i < runner.len_train - 1:
                    train_auc_list.append(auc)
                    train_acc_list.append(ap)
                elif i < runner.len_train + runner.len_val - 1:
                    val_auc_list.append(auc)
                    val_acc_list.append(ap)
                else:
                    test_auc_list.append(auc)
                    test_acc_list.append(ap)

        all_train_auc.append(np.mean(train_auc_list))
        all_val_auc.append(np.mean(val_auc_list))
        all_test_auc.append(np.mean(test_auc_list))

        all_train_acc.append(np.mean(train_acc_list))
        all_val_acc.append(np.mean(val_acc_list))
        all_test_acc.append(np.mean(test_acc_list))
        print('train_auc:', all_train_auc, 'val_auc:', all_val_auc, 'test_auc:', 'all_test_auc', all_test_auc)
        print('train_acc:', all_train_acc, 'val_acc:', all_val_acc, 'test_acc:', 'all_test_acc', all_test_acc)

    results = {
        # AUC
        "train_auc_mean": np.mean(all_train_auc),
        "train_auc_std": np.std(all_train_auc),
        "val_auc_mean": np.mean(all_val_auc),
        "val_auc_std": np.std(all_val_auc),
        "test_auc_mean": np.mean(all_test_auc),
        "test_auc_std": np.std(all_test_auc),

        # ACC
        "train_acc_mean": np.mean(all_train_acc),
        "train_acc_std": np.std(all_train_acc),
        "val_acc_mean": np.mean(all_val_acc),
        "val_acc_std": np.std(all_val_acc),
        "test_acc_mean": np.mean(all_test_acc),
        "test_acc_std": np.std(all_test_acc),
    }

    df = pd.DataFrame([results])
    return results


def train_evaluate(embeddings, train_data, ix, runner):

    train_auc_list = []
    val_auc_list = []
    test_auc_list = []
    _, pos_edge, neg_edge = prepare(train_data, ix + 1)[:3]
    auc, ac = runner.loss.predict(embeddings, pos_edge, neg_edge, runner.model.encoder.edge_decoder)

    return auc

