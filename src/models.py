import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from utils import *


class BundleMage(nn.Module):
    """
    Class of BundleMage model
    """
    def __init__(self, n_item, emb_dim,
                 lr=1e-3, decay=1e-5, batch_size=512,
                 drop=0.3, num_pos=2):
        """
        Class initializer
        """
        super(BundleMage, self).__init__()
        self.item_embs_mat = nn.Embedding(n_item, int(emb_dim/2))
        self.item_embs_gen = nn.Embedding(n_item, int(emb_dim/2))
        self.item_embs_shared = nn.Embedding(n_item, int(emb_dim/2))
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Linear(emb_dim, int(emb_dim / 2))
        self.w3 = nn.Linear(emb_dim, int(emb_dim / 2))
        mat_dim = [emb_dim, int(emb_dim/2), emb_dim]
        self.fnn1 = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in
                                   zip(mat_dim[:-1], mat_dim[1:])])
        gen_dim = [emb_dim, int(emb_dim/2), emb_dim]
        self.fnn2 = nn.ModuleList([nn.Linear(d_in, d_out) for
                                            d_in, d_out in
                                   zip(gen_dim[:-1], gen_dim[1:])])
        self.drop = nn.Dropout(drop)
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.init_weights()
        self.n_user = None
        self.n_item = None
        self.n_bundle = None
        self.bundle_item = None
        self.user_item = None
        self.user_bundle_trn = None
        self.user_bundle_test = None
        self.test_user_idx = None

        self.optimizer = optim.Adam(self.parameters(), lr=lr,
                                    weight_decay=decay)

    def get_mat_dataset(self, n_user, n_item, n_bundle, bundle_item, user_item,
                    user_bundle_trn, user_bundle_vld, user_bundle_test,
                    vld_user_idx, test_user_idx):
        """
        Get matching datasets
        """
        self.n_user = n_user
        self.n_item = n_item
        self.n_bundle = n_bundle
        self.bundle_item = bundle_item
        self.user_item = user_item
        self.user_bundle_trn = user_bundle_trn
        self.user_bundle_vld = user_bundle_vld
        self.user_bundle_test = user_bundle_test
        self.vld_user_idx = vld_user_idx
        self.test_user_idx = test_user_idx

    def get_gen_dataset(self, gen_input, gen_result, gen_user_bundle):
        """
        Get generation datasets
        """
        self.gen_input = gen_input
        self.gen_result = gen_result
        self.gen_user_bundle = gen_user_bundle

    def get_ratio(self):
        """
        Return masking ratio
        """
        return 0.5

    def init_weights(self):
        """
        Initialize parameters
        """
        nn.init.xavier_normal_(self.item_embs_mat.weight)
        nn.init.xavier_normal_(self.item_embs_gen.weight)
        nn.init.xavier_normal_(self.item_embs_shared.weight)
        for layer in self.fnn1:
            # Xavier initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal initialization for biases
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.fnn2:
            # Xavier initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal initialization for biases
            layer.bias.data.normal_(0.0, 0.001)
        for layer in [self.w1, self.w2, self.w3]:
            # Xavier initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal initialization for biases
            layer.bias.data.normal_(0.0, 0.001)


    def get_item_embs_mat(self):
        """
        Return item embeddings for bundle matching module
        """
        return torch.cat(
            (self.item_embs_shared.weight, self.item_embs_mat.weight), dim=1)

    def get_item_embs_gen(self):
        """
        Return item embeddings for bundle generation module
        """
        return torch.cat(
            (self.item_embs_shared.weight, self.item_embs_gen.weight), dim=1)

    def aggregation(self, mat, embs):
        """
        Aggregate embeddings using the given matrix
        """
        mat = normalize(mat, norm='l1', axis=1)
        mat = spy_sparse2torch_sparse(mat).to(DEVICE)
        embs_out = torch.sparse.mm(mat, embs)
        return embs_out

    def forward_gen(self, ui_batch, ub_batch, b_batch):
        """
        Predict bundle generation
        """
        bundle_embs = self.bundle_aggregate(self.bundle_item, self.get_item_embs_mat())
        user_batch = self.mat_encode(ui_batch, ub_batch, bundle_embs)
        bundle_batch = self.bundle_aggregate(b_batch, self.get_item_embs_gen())
        ub_z = self.gen_encode(user_batch, bundle_batch)
        b_recon = self.gen_decode(ub_z, self.get_item_embs_gen())
        return b_recon

    def gen_encode(self, user_batch, bundle_batch):
        """
        Encoder of bundle generation module
        """
        user_batch = self.w2(user_batch)
        bundle_batch = self.w3(bundle_batch)
        h = torch.cat((user_batch, bundle_batch), dim=1)
        for i, layer in enumerate(self.fnn2):
            h = layer(h)
            if i != len(self.fnn1) - 1:
                h = torch.relu(h)
        ub_z = h
        return self.drop(ub_z)

    def gen_decode(self, ub_z, item_embs):
        """
        Decoder of bundle generation module
        """
        b_recon = torch.matmul(ub_z, item_embs.T)
        return b_recon

    def forward_mat(self, ui_batch, ub_batch):
        """
        Predict bundle matching
        """
        bundle_embs = self.bundle_aggregate(self.bundle_item, self.get_item_embs_mat())
        u_z = self.mat_encode(ui_batch, ub_batch, bundle_embs)
        out = self.mat_decode(u_z, bundle_embs)
        return out

    def gated_mix(self, ui_h, ub_h):
        """
        Adaptive gated preference mixture
        """
        h_gate = self.w1(torch.cat((ui_h, ub_h), dim=1))
        gate = torch.sigmoid(h_gate)
        gated_pref = gate * ui_h + (torch.ones_like(gate).to(DEVICE) - gate) * ub_h
        h = gated_pref
        for i, layer in enumerate(self.fnn1):
            h = layer(h)
            if i != len(self.fnn1) - 1:
                h = torch.relu(h)
        u_z = h
        return u_z

    def mat_encode(self, ui_batch, ub_batch, bundle_embs):
        """
        Encoder of bundle matching module
        """
        ub_h = self.bundle_inter_aggregate(ub_batch, bundle_embs)
        ui_h = self.item_inter_aggregate(ui_batch, self.get_item_embs_mat())
        u_z = self.gated_mix(ui_h, ub_h)
        return self.drop(u_z)

    def mat_decode(self, u_z, bundle_embs):
        """
        Decoder of bundle matching module
        """
        ub_recon = torch.matmul(u_z, bundle_embs.T)
        return ub_recon

    def item_inter_aggregate(self, ui_batch, item_embs):
        """
        Aggregate item interactions
        """
        ui_h = self.aggregation(ui_batch, item_embs)
        return ui_h

    def bundle_inter_aggregate(self, ub_batch, bundle_embs):
        """
        Aggregate bundle interactions
        """
        ub_h = self.aggregation(ub_batch, bundle_embs)
        return ub_h

    def eval_loss(self, truth, recon):
        """
        Evaluate loss
        """
        truth = naive_sparse2tensor(truth).to(DEVICE)
        # BCE loss
        loss = -torch.mean(
            torch.sum(F.log_softmax(recon, 1) * truth, -1))
        return loss

    def bundle_aggregate(self, bundle_item_mat, item_embs):
        """
        Aggregate item embeddings for bundles
        """
        mat = normalize(bundle_item_mat, norm='l1', axis=1)
        mat = spy_sparse2torch_sparse(mat).to(DEVICE)
        bundle_embs = torch.sparse.mm(mat, item_embs)
        return bundle_embs

    def split_csr(self, csr_mat, ratio=0.5):
        """
        Randomly split csr matrix
        """
        num_nonzero = len(csr_mat.nonzero()[0])
        total_idx = np.random.permutation(num_nonzero)
        split_idx = int(num_nonzero * ratio)
        idx1, idx2 = total_idx[:split_idx], total_idx[split_idx:]
        row1, col1 = csr_mat.nonzero()[0][idx1], csr_mat.nonzero()[1][idx1]
        data1 = np.ones_like(row1)
        row2, col2 = csr_mat.nonzero()[0][idx2], csr_mat.nonzero()[1][idx2]
        data2 = np.ones_like(row2)
        mat1 = csr_matrix((data1, (row1, col1)), shape=csr_mat.shape)
        mat2 = csr_matrix((data2, (row2, col2)), shape=csr_mat.shape)
        return mat1, mat2

    def update_model(self):
        """
        Update the model
        """
        self.train()
        ratio = self.get_ratio()

        # Matching
        assert self.user_bundle_trn.shape[0] == self.user_item.shape[0]
        num_trn = self.user_bundle_trn.shape[0]
        idx_list = list(range(num_trn))
        trn_rec_loss = 0.
        np.random.shuffle(idx_list)
        for batch_idx, start_idx in enumerate(range(0, num_trn, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_trn)
            self.optimizer.zero_grad()
            ui_batch = self.user_item[idx_list[start_idx:end_idx]]
            ub_batch = self.user_bundle_trn[idx_list[start_idx:end_idx]]
            in_ub_batch, out_ub_batch = self.split_csr(ub_batch, ratio=ratio)
            ub_recon = self.forward_mat(ui_batch, in_ub_batch)
            loss = self.eval_loss(ub_batch, ub_recon)
            loss.backward()
            trn_rec_loss += loss.item()
            self.optimizer.step()
        trn_rec_loss = trn_rec_loss / (batch_idx + 1)
        trn_loss = trn_rec_loss

        # Generation
        ub_trn_coo = self.user_bundle_trn.tocoo()
        users, bundles = ub_trn_coo.row, ub_trn_coo.col
        user_bundle = np.concatenate((users[:, np.newaxis], bundles[:, np.newaxis]), axis=1)
        num_trn = user_bundle.shape[0]
        idx_list = list(range(num_trn))
        trn_gen_loss = 0.
        np.random.shuffle(idx_list)
        for batch_idx, start_idx in enumerate(range(0, num_trn, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_trn)
            ub_idx_batch = user_bundle[idx_list[start_idx:end_idx]]
            self.optimizer.zero_grad()
            ui_batch = self.user_item[ub_idx_batch[:, 0]]
            ub_batch = self.user_bundle_trn[ub_idx_batch[:, 0]]
            in_ub_batch, _ = self.split_csr(ub_batch, ratio=ratio)
            b_batch = self.bundle_item[ub_idx_batch[:, 1]]
            in_b_batch, out_b_batch = self.split_csr(b_batch, ratio=ratio)
            b_recon = self.forward_gen(ui_batch, in_ub_batch, in_b_batch)
            loss = self.eval_loss(b_batch, b_recon)
            loss.backward()
            trn_gen_loss += loss.item()
            self.optimizer.step()
        trn_gen_loss = trn_gen_loss / (batch_idx + 1)
        trn_loss = trn_loss + trn_gen_loss

        return trn_loss

    def evaluate_mat(self, ks):
        """
        Evaluate bundle matching module
        """
        self.eval()
        with torch.no_grad():
            batch_size = 1000
            vld_loss_list = []
            vld_recall_list, vld_ndcg_list = [], []
            ui_arr = self.user_item[self.vld_user_idx]
            ub_arr = self.user_bundle_trn[self.vld_user_idx]
            for batch_idx, start_idx in enumerate(range(0, ui_arr.shape[0], batch_size)):
                end_idx = min(start_idx + batch_size, ui_arr.shape[0])
                ui_batch = ui_arr[start_idx:end_idx]
                ub_batch = ub_arr[start_idx:end_idx]
                ub_recon = self.forward_mat(ui_batch, ub_batch)
                vld_loss = self.eval_loss(ub_batch, ub_recon)
                vld_recon_gather = \
                    torch.gather(ub_recon, dim=1,
                                 index=torch.LongTensor(
                                     self.user_bundle_vld[start_idx:end_idx]).to(DEVICE))
                vld_recalls, vld_ndcgs = evaluate_accuracies(vld_recon_gather,
                                                             ks=ks, batch=True)
                vld_loss_list.append(vld_loss)
                vld_recall_list.append(vld_recalls)
                vld_ndcg_list.append(vld_ndcgs)
            vld_loss = torch.FloatTensor(vld_loss_list).mean()
            vld_recalls = list(np.array(vld_recall_list).sum(axis=0)/ui_arr.shape[0])
            vld_ndcgs = list(np.array(vld_ndcg_list).sum(axis=0)/ui_arr.shape[0])

            test_loss_list = []
            test_recall_list, test_ndcg_list = [], []
            ui_arr = self.user_item[self.test_user_idx]
            ub_arr = self.user_bundle_trn[self.test_user_idx]
            for batch_idx, start_idx in enumerate(range(0, ui_arr.shape[0], batch_size)):
                end_idx = min(start_idx + batch_size, ui_arr.shape[0])
                ui_batch = ui_arr[start_idx:end_idx]
                ub_batch = ub_arr[start_idx:end_idx]
                ub_recon = self.forward_mat(ui_batch, ub_batch)
                test_loss = self.eval_loss(ub_batch, ub_recon)
                test_recon_gather = \
                    torch.gather(ub_recon, dim=1,
                                 index=torch.LongTensor(
                                     self.user_bundle_test[start_idx:end_idx]).to(DEVICE))
                test_recalls, test_ndcgs = evaluate_accuracies(test_recon_gather,
                                                               ks=ks, batch=True)
                test_loss_list.append(test_loss)
                test_recall_list.append(test_recalls)
                test_ndcg_list.append(test_ndcgs)
            test_loss = torch.FloatTensor(test_loss_list).mean()
            test_recalls = list(np.array(test_recall_list).sum(axis=0)/ui_arr.shape[0])
            test_ndcgs = list(np.array(test_ndcg_list).sum(axis=0)/ui_arr.shape[0])
        return vld_loss, test_loss, vld_recalls, vld_ndcgs, test_recalls, test_ndcgs

    def evaluate_gen(self, ks):
        """
        Evaluate bundle generation module
        """
        self.eval()
        with torch.no_grad():
            ub_trn_coo = self.gen_user_bundle.tocoo()
            users, bundles = ub_trn_coo.row, ub_trn_coo.col
            user_bundle = np.concatenate((users[:, np.newaxis], bundles[:, np.newaxis]), axis=1)
            num_gen = user_bundle.shape[0]
            batch_size = 1000
            test_recall_list, test_ndcg_list = [], []

            for batch_idx, start_idx in enumerate(range(0, num_gen, batch_size)):
                end_idx = min(start_idx + batch_size, num_gen)
                ub_idx_batch = user_bundle[start_idx:end_idx]
                ui_batch = self.user_item[ub_idx_batch[:, 0]]
                ub_batch = self.user_bundle_trn[ub_idx_batch[:, 0]]
                b_batch = self.gen_input[ub_idx_batch[:, 1]]
                b_recon = self.forward_gen(ui_batch, ub_batch, b_batch)
                test_recon_gather = \
                    torch.gather(b_recon, dim=1,
                                 index=torch.LongTensor(
                                     self.gen_result[ub_idx_batch[:, 1]]).to(DEVICE))

                true_score = torch.zeros(test_recon_gather.shape)
                true_score[:, :self.num_pos] = 1
                test_recalls, test_ndcgs = evaluate_gen_accruacies(
                    test_recon_gather, true_score, ks=ks)

                test_recall_list.append(test_recalls)
                test_ndcg_list.append(test_ndcgs)

            test_recalls = list(np.array(test_recall_list).sum(axis=0)/num_gen)
            test_ndcgs = list(np.array(test_ndcg_list).sum(axis=0)/num_gen)

        return test_recalls, test_ndcgs
