import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
# from torch.cuda.amp import GradScaler
from torch.cuda.amp import GradScaler
import math
from scipy.stats import norm



class CustomDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=64, dropout=0.1):

        super(CustomDecoderLayer, self).__init__()

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, tgt_mask=None):
        # Self-attention layer
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feed-forward layer
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
    


class CustomTransformerDecoder(nn.Module):

    def __init__(self, d_model, num_layers, dim_feedforward=64, dropout=0.1):

        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([CustomDecoderLayer(d_model, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers


    def forward(self, tgt, tgt_mask=None):

        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask)

        return output
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)
    


class Actor(nn.Module):

    def __init__(self, device, params, des_space, comp_list, num_objectives=5):
        super(Actor, self).__init__()
        self.device = device
        self.params = params
        self.des_space = des_space
        self.num_objectives = num_objectives
        self.comp_list = comp_list
        self.clip_ratio = params['clip_ratio']
        self.dense_dim = 16

        # Shared Backbone
        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=0.1
        )

        # Multi-Objective Heads: One list of output layers per objective
        self.heads = nn.ModuleList()
        for _ in range(num_objectives):
            head_layers = nn.ModuleList()
            for var in des_space:
                if var['type'] == 'continuous':
                    head_layers.append(nn.Linear(self.dense_dim, 2))
                elif var['type'] == 'discrete':
                    head_layers.append(nn.Linear(self.dense_dim, len(var['range'])))
            self.heads.append(head_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scaler = GradScaler()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)


    def generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask
    

    def get_valid_panel_mask(self, observation_tensor, des_space, ind):

        """
        Vectorized version: builds a (batch_size, num_panels) boolean mask.
        
        observation_tensor: (batch_size, seq_len) tensor of actions so far
                            seq_len must be at least enough to contain structure_id (0) and shelves (4)
        des_space: design space definition
        Returns: mask of shape (batch_size, vocab_size) where vocab_size = len(des_space[5]['range'])
        """
        batch_size = observation_tensor.size(0)
        num_panel_options = len(des_space[5]['range'])
        structure_ids = observation_tensor[:, 0].long()
        shelves = observation_tensor[:, 4].long()
        base_panels_lookup = torch.tensor([5, 6, 8], device=self.device)
        base_panels = base_panels_lookup[structure_ids]

        mask = torch.zeros((batch_size, num_panel_options), dtype=torch.bool, device=self.device)
        
        # Set base panel validity
        if self.comp_list[ind//5 - 1].pointing:
            for b in range(batch_size):
                mask[b, 0:base_panels[b]] = True
        else:
            for b in range(batch_size):
                mask[b, 0:base_panels[b]] = True
                mask[b, 8:shelves[b]+8] = True  # shelves start at index 8
                mask[b, 11:shelves[b]+11] = True  # shelves have two sides

        return mask  # (batch_size, vocab_size) boolean

    
    def forward(self, x, var_ind, obj_ind):

        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = x.unsqueeze(-1)
            x = self.encoder(x)
            x = self.positional_encoding(x)
            mask = self.generate_square_subsequent_mask(x.size(1)).to(self.device) # x.size(1) is the sequence length
            x = self.transformer_decoder(x, tgt_mask=mask)
            # Objective-specific head
            x = self.heads[obj_ind][var_ind](x)
            if self.des_space[var_ind]['type'] == 'continuous':
                x = torch.exp(x)
            elif self.des_space[var_ind]['type'] == 'discrete':
                x = F.softmax(x, dim=-1)
            else:
                print("INVALID DESIGN SPACE")

        return x
    

    def sample_action(self, observation, var_ind, obj_ind):

        if len(observation[0]) == 0:
            observation = [[0] for x in range(len(observation))]
        input_observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if var_ind >= len(self.des_space):
            out_head = ((var_ind-len(self.des_space)) % 5) + 5
        else:
            out_head = var_ind
        output = self.forward(input_observation, out_head, obj_ind)
        output_last = output[:, -1, :]

        if self.des_space[out_head]['type'] == 'continuous':
            alpha = output_last[:, 0].squeeze()
            beta = output_last[:, 1].squeeze()
            action_dist = torch.distributions.Beta(alpha, beta)
            action_ids = action_dist.sample().squeeze()
            action_probs = action_dist.log_prob(action_ids)
        elif self.des_space[out_head]['type'] == 'discrete':
            log_probs = torch.log(output_last + 1e-10)
            # samples = torch.distributions.categorical.Categorical(logits=log_probs).sample()
            if var_ind == 5 or (var_ind > 5 and (var_ind - 5) % 5 == 0):  # panel choice indices
                valid_mask = self.get_valid_panel_mask(input_observation, self.des_space, var_ind).to(self.device)
                log_probs[~valid_mask] = -1e9
                dist = torch.distributions.categorical.Categorical(logits=log_probs)
            else:
                dist = torch.distributions.categorical.Categorical(logits=log_probs)
            samples = dist.sample()
            action_ids = samples.squeeze()
            action_probs = log_probs.gather(1, action_ids.unsqueeze(-1)).squeeze()

        return action_probs, action_ids
    

    def ppo_update(self, observation, action, logprob, advantage, obj_ind):

        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            num_vars = observation.size(1)
            new_log_probs = torch.zeros_like(action, dtype=torch.float32).to(self.device)
            pointing_comps = torch.tensor(
                [comp.pointing for comp in self.comp_list] * self.params['mini_batch_size'],
                dtype=torch.bool,
                device=self.device
            )
            # Find the index of the first pointing and first nonpointing component
            first_pointing_idx = next((i for i, p in enumerate(pointing_comps) if p), None)
            first_nonpointing_idx = next((i for i, p in enumerate(pointing_comps) if not p), None)
            for i in range(len(self.des_space)):
                var_mask = torch.zeros((observation.size(0)), dtype=torch.bool, device=self.device)
                if i < 5:
                    var_mask[i::num_vars] = True
                else:
                    var_mask[i::5] = True
                    var_mask[i-5::num_vars] = False
                var_observation = observation[var_mask]
                var_action = action[var_mask]
                # need to separate discrete and continuous to calculate logprobs
                if self.des_space[i]['type'] == 'continuous':
                    params = self.forward(var_observation, i, obj_ind)
                    alpha = params[:,-1,0].squeeze()
                    beta = params[:,-1,1].squeeze()
                    action_dist = torch.distributions.Beta(alpha, beta)
                    var_new_logprobs = action_dist.log_prob(var_action)
                elif self.des_space[i]['type'] == 'discrete':
                    pred_probs = self.forward(var_observation, i, obj_ind)[:,-1,:]
                    log_probs = torch.log(pred_probs + 1e-10)
                    if i == 5 or (i > 5 and (i - 5) % 5 == 0):  # panel choice indices
                        # pointing and nonpointing components have different valid panel choices
                        valid_mask_pointing = self.get_valid_panel_mask(var_observation[pointing_comps], self.des_space, first_pointing_idx).to(self.device)
                        valid_mask_nonpointing = self.get_valid_panel_mask(var_observation[~pointing_comps], self.des_space, first_nonpointing_idx).to(self.device)
                        valid_mask = torch.zeros((var_observation.size(0), log_probs.size(1)), dtype=torch.bool, device=self.device)
                        valid_mask[pointing_comps] = valid_mask_pointing
                        valid_mask[~pointing_comps] = valid_mask_nonpointing
                        log_probs[~valid_mask] = -1e9
                    var_new_logprobs = torch.sum(torch.mul(F.one_hot(var_action.long(), num_classes=len(self.des_space[i]['range'])), log_probs), dim=-1)
                else:
                    print("INVALID DESIGN SPACE")

                new_log_probs[var_mask] = var_new_logprobs

            new_log_probs = new_log_probs.view(-1)

            ratio = torch.exp(new_log_probs - logprob)
            min_advantage = torch.where(
                advantage > 0,
                (1 + self.clip_ratio) * advantage,
                (1 - self.clip_ratio) * advantage
            )
            policy_loss = -torch.mean(torch.min(ratio * torch.t(advantage), min_advantage))

        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        kl = torch.mean(new_log_probs - logprob)

        return policy_loss.item(), kl.item()
    


class Critic(nn.Module):

    def __init__(self, device, params, num_objectives, input_dim):

        super(Critic, self).__init__()
        self.device = device
        self.params = params

        self.nhead = 2
        self.dense_dim = 16
        self.scaler = GradScaler()
        self.num_objectives = num_objectives
        self.input_dim = input_dim

        self.input_layer = nn.Linear(self.input_dim, self.dense_dim)
        self.hidden_layer = nn.Linear(self.dense_dim, self.dense_dim)
        self.output_layer = nn.Linear(self.dense_dim, self.num_objectives)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    

    def forward(self, x):

        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = self.input_layer(x)
            x = self.hidden_layer(x)
            x = self.output_layer(x)

        return x
    

    def sample_critic(self, observation):

        input_observations = []
        for obs in observation:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.input_dim:
                input_obs.extend([0])
            input_observations.append(input_obs)
        input_observations = torch.tensor(input_observations, dtype=torch.float32, device=self.device)
        output = self(input_observations)  # shape: (num components, 5)

        return output
    

    def ppo_update(self, observation, return_tensor, obj_ind):

        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            new_pred_values = self(observation)
            new_pred_reward = new_pred_values[:, obj_ind]

            value_loss = torch.mean((new_pred_reward - return_tensor) ** 2)

        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss.item()
