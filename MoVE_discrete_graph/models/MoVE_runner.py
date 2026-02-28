import os
import sys

from models.utils.mutils import *
from models.utils.inits import prepare
from models.utils.loss import EnvLoss
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class Runner(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        if args.dataset in ['collab_04', 'collab_06', 'collab_08']:
            self.n_nodes = data["x"].shape[1]
        else:
            self.n_nodes = data["x"].shape[0]

        self.writer = writer
        if args.dataset in ['aminer']:
            self.len = args.len_train
        else:
            self.len = len(data["train"]["edge_index_list"])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.criterion = torch.nn.CrossEntropyLoss()

        x = data["x"].to(args.device).clone().detach()
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x

        self.loss = EnvLoss(args)
        print("total length: {}, test length: {}".format(self.len, args.testlength))

        if args.dataset == 'aminer':
            self.edge_index_list_pre = [
                data["edge_index"][ix].long().to(args.device)
                for ix in range(self.len)]
        else:
            self.edge_index_list_pre = [
                data["train"]["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ]

    def cal_y(self, embeddings, decoder, edge_index, device):
        preds = torch.tensor([]).to(device)
        z = embeddings
        pred = decoder(z, edge_index)
        preds = torch.cat([preds, pred])
        return preds

    def classification_cal_y(self, embeddings, decoder, node_masks, device, ix):
        z = embeddings
        mask = node_masks[ix]
        pred = decoder(z)[mask]
        return pred

    def accuracy(self, y, label):
        _, predicted = torch.max(y, 1)
        correct = (predicted == label).sum().item()
        total = label.size(0)
        acc = correct / total
        return acc

    def cal_loss(self, y, label):
        criterion = torch.nn.BCELoss()
        return criterion(y, label)

    def vgae_test(self, epoch, data, encoder):
        args = self.args
        train_auc_list = []
        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        for i in range(self.len):

            embeddings, _, _ = self.model.encode(self.x[i], data['edge_index_list'][i].long().to(args.device))

            if i < self.len - 1:
                z = embeddings
                pos_index = data["pedges"][i]  # torch edge index
                neg_index = data["nedges"][i]
                # edge_index, pos_edge, neg_edge = prepare(pos_index, neg_index)[:3]
                edge_index, pos_edge, neg_edge = prepare(data, i+1)[:3]

                if is_empty_edges(neg_edge):
                    continue
                auc, ap = self.loss.predict(z, pos_edge, neg_edge, encoder)
                if i < self.len_train - 1:
                    train_auc_list.append(auc)
                elif i < self.len_train + self.len_val - 1:
                    val_auc_list.append(auc)
                else:
                    test_auc_list.append(auc)
        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]
