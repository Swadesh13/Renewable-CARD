import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, config):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        data_dim = config.model.y_dim
        self.window_size = config.data.window_size
        self.dim_x = config.model.x_dim
        self.dim_per_window = int(config.model.x_dim / config.data.window_size)
        if self.cat_x:
            data_dim += config.model.x_dim
        if self.cat_y_pred:
            data_dim += config.model.y_dim
        self.lin1 = ConditionalLinear(data_dim, config.model.feature_dim, n_steps)
        self.lin2 = ConditionalLinear(config.model.feature_dim, config.model.feature_dim, n_steps)
        self.lin3 = ConditionalLinear(config.model.feature_dim, config.model.feature_dim, n_steps)
        self.lin4 = nn.Linear(config.model.feature_dim, 1)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        # eps_pred = F.softplus(self.lin3_(eps_pred))
        return self.lin4(eps_pred)


# deterministic feed forward neural network
class DeterministicFeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        hid_layers,
        use_batchnorm=False,
        use_layernorm=False,
        negative_slope=0.01,
        dropout_rate=0,
        window_size=5,
        lstm=True,
    ):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.window_size = window_size
        self.dim_per_window = int(dim_in / self.window_size)
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.use_layernorm = use_layernorm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)
        self.lstm = lstm
        if self.lstm:
            self.lstm = nn.LSTM(self.dim_per_window, self.dim_per_window, 2, batch_first=True, dropout=0.1)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            elif self.use_layernorm:
                layers.append(nn.LayerNorm(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        if self.lstm:
            x, _ = self.lstm(x.reshape(-1, self.window_size, self.dim_per_window))
            x = x.reshape(-1, self.dim_in)
        return self.network(x)


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                            shall be a small positive value.
                            Default: 0
            best_score: value of the best metric on the validation set.
            best_epoch: epoch with the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):
        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0
