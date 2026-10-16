"""PyTorch MLP with learned embeddings for categorical features."""
import copy

import numpy as np
import torch
import torch.nn as nn

from taxi.features import LOW_CAT, ZONE, numeric_cols

CARDINALITY = {"PULocationID": 266, "DOLocationID": 266, "hour": 24, "dow": 7, "ratecode": 7, "vendor": 8}


class TabularMLP(nn.Module):
    def __init__(self, n_num, cat_cols, emb_dim, hidden, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(CARDINALITY[c], min(emb_dim, CARDINALITY[c])) for c in cat_cols])
        in_dim = n_num + sum(e.embedding_dim for e in self.embs)
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat([x_num] + embedded, dim=1)).squeeze(1)


class MLPModel:
    def __init__(self, task, cfg, seed=0):
        self.task, self.cfg, self.seed = task, cfg, seed
        self.num = numeric_cols(task)
        self.cat = ZONE + LOW_CAT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tensors(self, X):
        xn = (X[self.num].to_numpy(np.float32) - self.mean) / self.std
        xc = X[self.cat].to_numpy(np.int64)
        return torch.from_numpy(xn), torch.from_numpy(xc)

    def _loss(self, logits, y):
        if self.task == "duration":
            return nn.functional.mse_loss(logits, y)
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    @torch.no_grad()
    def _raw(self, xn, xc, bs=65536):
        self.net.eval()
        out = [self.net(xn[i:i + bs].to(self.device), xc[i:i + bs].to(self.device)).float().cpu()
               for i in range(0, len(xn), bs)]
        return torch.cat(out)

    def fit(self, X, y, Xva, yva):
        torch.manual_seed(self.seed)
        self.mean = X[self.num].to_numpy(np.float32).mean(0)
        self.std = X[self.num].to_numpy(np.float32).std(0) + 1e-6
        xn, xc = self._tensors(X)
        vn, vc = self._tensors(Xva)
        yt = torch.as_tensor(y, dtype=torch.float32)
        yv = torch.as_tensor(yva, dtype=torch.float32)

        c = self.cfg
        self.net = TabularMLP(len(self.num), self.cat, c["emb_dim"], c["hidden"], c["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"])
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(c["epochs"]):
            self.net.train()
            perm = torch.randperm(len(xn))
            for i in range(0, len(xn), c["batch_size"]):
                idx = perm[i:i + c["batch_size"]]
                logits = self.net(xn[idx].to(self.device), xc[idx].to(self.device))
                loss = self._loss(logits, yt[idx].to(self.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            val_loss = self._loss(self._raw(vn, vc), yv).item()
            print(f"    mlp epoch {epoch + 1}  val loss {val_loss:.4f}")
            if val_loss < best:
                best, best_state, bad = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        out = self._raw(*self._tensors(X))
        return (out if self.task == "duration" else torch.sigmoid(out)).numpy()
