import torch
import torch.nn as nn
import torch.optim as optim


class Item2Vec(nn.Module):
    def __init__(self, item_len, embed_dim, epoch, learning_rate, early_stop):
        super(Item2Vec, self).__init__()
        self.item_len = item_len
        self.embed_layer = embed_dim
        self.embedding_layer = nn.Embedding(self.item_len, self.embed_layer)
        self.activation = nn.Sigmoid()
        self.epoch = epoch
        self.lr = learning_rate
        self.early_stop = early_stop

    def forward(self, target_i, context_j):
        target_emb = self.embed_layer(target_i)
        context_emb = self.embed_layer(context_j)
        output = torch.sum(target_emb * context_emb, dim=1)
        output = self.activation(output)

        return output.view(-1)

    def fit(self, train_loader):
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = optim.Adam(self.parameters(), self.lr)
        current_loss = 0.0
        last_loss = 0.0
        for ep in range(1, self.epoch + 1):
            for target_i, context_j, label in train_loader:
                pred = self.forward(target_i, context_j)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                delta_loss = float(current_loss - last_loss)
                if abs(delta_loss) < 1e-5 and self.early_stop:
                    break
                else:
                    last_loss = current_loss
