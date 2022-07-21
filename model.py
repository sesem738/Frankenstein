import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import model_transformer as mt
    
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
NHEAD = 4
NLAYERS = 2
DROPOUT = 0.2

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Critic
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, model=None):
        super(QNetwork, self).__init__()

        self.model = model

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

        if self.model in ["BERT", "GTrXL"]:
            # Continuous embedding
            self.embedding = nn.Linear(num_inputs + num_actions, 256)
            self.linear1 = nn.Linear(256, 256)
            self.linear4 = nn.Linear(256, 256)

            # Init transformer
            # self.transformer = BERT_Torch(ntoken=256, d_model=256, nhead=4, d_hid=256*4, nlayers=2, dropout=0.2)
            # self.transformer = GTrXL(256, NHEAD, NLAYERS)
            self.transformer = getattr(mt, self.model)(ntoken=256, d_model=256, nhead=NHEAD, d_hid=256*4, nlayers=NLAYERS, dropout=DROPOUT)

        self.apply(weights_init_)

    def forward(self, state, action):
        if self.model not in ["BERT", "GTrXL"]:
            state = state[:,[0],:]
            action = action[:,[0],:]

        xu = torch.cat([state, action], -1)

        if self.model in ["BERT", "GTrXL"]:
            # Continuous embedding
            xu = self.embedding(xu)
            # xu_past, xu_current = xu[:-1, :], xu[-1, :].expand(1,-1)
            # print("*********************************************************")
            # print(xu_past.shape, xu_current.shape)
            # Transformer here
            # xu_past = self.transformer(xu_past)
            xu = self.transformer(xu)
            # print(xu_past.shape, xu_current.shape)
            # print(xu_past,xu_current)
            # print("*********************************************************")
            # xu = torch.cat([xu_past, xu_current], 0)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Actor 
class GaussianPolicy(nn.Module):
    
    def __init__(self, num_inputs, num_actions, action_space=None, model=None):
        super(GaussianPolicy, self).__init__()

        self.model = model


        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)


        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.Tensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.Tensor(
                (action_space.high + action_space.low) / 2.)

        if self.model in ["BERT", "GTrXL"]:
            # Continuous embedding
            self.embedding = nn.Linear(num_inputs, 256)
            self.linear1 = nn.Linear(256, 256)

            # import transformer
            # self.transformer = GTrXL(256, NHEAD, NLAYERS)
            self.transformer = getattr(mt, self.model)(ntoken=256, d_model=256, nhead=NHEAD, d_hid=256*4, nlayers=NLAYERS, dropout=DROPOUT)

        self.apply(weights_init_)
    def forward(self, state):

        if self.model in ["BERT", "GTrXL"]:
            # Continuous embedding
            # print(state.shape)
            state = self.embedding(state)
            # state_past, state_current = state[:-1, :], state[-1, :].expand(1,-1)

            # Transformer here
            # state_past = self.transformer(state_past)
            state = self.transformer(state)
            # state = torch.cat([state_past, state_current], 0)
        else:
            try:
                state = state[:,[0],:]
            except:
                # print(state.shape)
                raise Exception("State post transformer shape error")

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean