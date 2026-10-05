import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
# from torch.cuda.amp import GradScaler
from torch.cuda.amp import GradScaler
import math
from scipy.stats import norm

from utils.ppo_utils import is_legacy, amp_settings, amp_ctx, add_positional_encoding, gather_positions, throttle
import time



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

    def __init__(self, d_model, dropout=0.1, max_len=20000):
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

    def __init__(self, device, params, des_space, comp_list, num_objectives):

        super(Actor, self).__init__()
        self.device = device
        self.params = params
        self.des_space = des_space
        self.comp_list = comp_list
        self.num_objectives = num_objectives

        self.nhead = 2
        self.dense_dim = 32
        self.legacy = is_legacy(params)
        self.use_amp, dropout = amp_settings(params)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_ratio = params['clip_ratio']

        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim, dropout=dropout)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=2,
            dim_feedforward=self.dense_dim,
            dropout=dropout
        )
        # self.output_layer = nn.Linear(self.dense_dim, self.num_actions)
        self.output_layers = nn.ModuleList()
        for var in des_space:
            if var['type'] == 'continuous':
                self.output_layers.append(nn.Linear(self.dense_dim, 2))
            elif var['type'] == 'discrete':
                self.output_layers.append(nn.Linear(self.dense_dim, len(var['range'])))
            else:
                print("INVALID DESIGN SPACE")


        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
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

    
    def forward(self, x, ind):

        with amp_ctx(self):
            x = x.unsqueeze(-1)
            x = self.encoder(x)
            x = add_positional_encoding(self.positional_encoding, x, self.legacy)
            mask = self.generate_square_subsequent_mask(x.size(1)).to(self.device) # x.size(1) is the sequence length
            x = self.transformer_decoder(x, tgt_mask=mask)
            x = self.output_layers[ind](x)
            if self.des_space[ind]['type'] == 'continuous':
                x = torch.exp(torch.clamp(x.float(), min=-10, max=10))
            elif self.des_space[ind]['type'] == 'discrete':
                x = F.softmax(x, dim=-1)
            else:
                print("INVALID DESIGN SPACE")

        return x
    

    def sample_action(self, observation, ind):

        if len(observation[0]) == 0:
            observation = [[0] for x in range(len(observation))]
        input_observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if ind >= len(self.des_space):
            out_head = ((ind-len(self.des_space)) % 5) + 5
        else:
            out_head = ind
        output = self.forward(input_observation, out_head)
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
            if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):  # panel choice indices
                valid_mask = self.get_valid_panel_mask(input_observation, self.des_space, ind).to(self.device)
                log_probs[~valid_mask] = -1e9
                dist = torch.distributions.categorical.Categorical(logits=log_probs)
            else:
                dist = torch.distributions.categorical.Categorical(logits=log_probs)
            samples = dist.sample()
            action_ids = samples.squeeze()
            if self.legacy:
                action_probs = log_probs.gather(1, action_ids.unsqueeze(-1)).squeeze()
            else:
                action_probs = dist.log_prob(samples).squeeze()

        return action_probs, action_ids
    

    def _head_logprobs(self, i, obs, act, pos, pointing, pointing_var, nonpointing_var):
        # log-probs of actions `act` under output head i
        if self.des_space[i]['type'] == 'continuous':
            params = gather_positions(self.forward(obs, i), pos)
            return torch.distributions.Beta(params[:, 0], params[:, 1]).log_prob(act)
        log_probs = torch.log(gather_positions(self.forward(obs, i), pos) + 1e-10)
        if i == 5 or (i > 5 and (i - 5) % 5 == 0):  # panel choice indices
            # pointing and nonpointing components have different valid panel choices
            valid_mask = torch.zeros((obs.size(0), log_probs.size(1)), dtype=torch.bool, device=self.device)
            if pointing.any():
                valid_mask[pointing] = self.get_valid_panel_mask(obs[pointing], self.des_space, pointing_var)
            if (~pointing).any():
                valid_mask[~pointing] = self.get_valid_panel_mask(obs[~pointing], self.des_space, nonpointing_var)
            log_probs = log_probs.masked_fill(~valid_mask, -1e9)
            if not self.legacy:
                # renormalize over valid panels like sample_action
                log_probs = F.log_softmax(log_probs, dim=-1)
        return log_probs.gather(1, act.long().unsqueeze(-1)).squeeze(-1)


    def ppo_update(self, observation, action, logprob, advantage, action_pos=None):
        # action_pos: position each action was sampled at (observations are zero padded)
        # The loss is a mean over rows, so it is accumulated over chunks of rows (same gradient).

        self.optimizer.zero_grad()
        n = observation.size(0)
        num_vars = observation.size(1) - self.num_objectives
        chunk = self.params.get('update_chunk', 2048)
        pointing_list = [comp.pointing for comp in self.comp_list]
        first_pointing_idx = pointing_list.index(True)
        first_nonpointing_idx = pointing_list.index(False)
        if not self.legacy:
            # get_valid_panel_mask expects a variable index, not a component index
            first_pointing_idx = 5 * (first_pointing_idx + 1)
            first_nonpointing_idx = 5 * (first_nonpointing_idx + 1)
        pointing_rows = torch.tensor(pointing_list * self.params['mini_batch_size'], dtype=torch.bool, device=self.device)

        policy_loss, kl = 0.0, 0.0
        for i in range(len(self.des_space)):
            var_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            if i < 5:
                var_mask[i::num_vars] = True
            else:
                var_mask[i::5] = True
                var_mask[i-5::num_vars] = False
            rows = torch.nonzero(var_mask).squeeze(-1)
            for s in range(0, len(rows), chunk):
                t_chunk = time.time()
                r = rows[s:s + chunk]
                pointing = pointing_rows[s:s + chunk] if i >= 5 else None
                with amp_ctx(self):
                    new_lp = self._head_logprobs(i, observation[r], action[r],
                                                 None if action_pos is None else action_pos[r],
                                                 pointing, first_pointing_idx, first_nonpointing_idx)
                    ratio = torch.exp(new_lp - logprob[r])
                    adv = advantage[r]
                    min_advantage = torch.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)
                    loss = -torch.sum(torch.min(ratio * adv, min_advantage)) / n
                self.scaler.scale(loss).backward()
                policy_loss += loss.item()
                kl += torch.sum(new_lp - logprob[r]).item() / n
                throttle(self.params, t_chunk, sync=True)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return policy_loss, kl
    

class Critic(nn.Module):

    def __init__(self, device, params, num_objectives):

        super(Critic, self).__init__()
        self.device = device
        self.params = params

        self.nhead = 2
        self.dense_dim = 16
        self.legacy = is_legacy(params)
        self.use_amp, dropout = amp_settings(params)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_ratio = params['clip_ratio']
        self.num_objectives = num_objectives

        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim, dropout=dropout)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            # nhead=self.nhead,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=dropout
        )
        self.output_layer = nn.Linear(self.dense_dim, self.num_objectives)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)


    def generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask
    

    def forward(self, x, pos=None):

        with amp_ctx(self):
            x = x.unsqueeze(-1)
            x = self.encoder(x)
            x = add_positional_encoding(self.positional_encoding, x, self.legacy)
            mask = self.generate_square_subsequent_mask(x.size(1)).to(self.device)
            x = self.transformer_decoder(x, tgt_mask=mask)
            x = self.output_layer(x)
            x = gather_positions(x, pos)

        return x
    

    def sample_critic(self, observation):
        
        input_observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        # Ensure shape is [batch, seq_len] regardless of input nesting
        if input_observation.dim() == 3 and input_observation.size(0) == 1:
            input_observation = input_observation.squeeze(0)
        elif input_observation.dim() == 3:
            # [batch, 1, seq_len] -> [batch, seq_len]
            input_observation = input_observation.squeeze(1)
        elif input_observation.dim() == 1:
            input_observation = input_observation.unsqueeze(0)
        output = self(input_observation)
        
        return output
    

    def ppo_update(self, observation, return_tensor, weights_tensor, pos=None):

        self.optimizer.zero_grad()
        # accumulate over chunks (same gradient as one batch, much less GPU memory)
        n = observation.size(0)
        chunk = self.params.get('update_chunk', 2048)
        value_loss = 0.0
        for s in range(0, n, chunk):
            t_chunk = time.time()
            sl = slice(s, s + chunk)
            with amp_ctx(self):
                new_pred_values = self(observation[sl], None if pos is None else pos[sl])
                new_pred_reward = torch.sum(-new_pred_values * weights_tensor[sl], dim=-1)
                loss = torch.sum((new_pred_reward - return_tensor[sl]) ** 2) / n
            self.scaler.scale(loss).backward()
            value_loss += loss.item()
            throttle(self.params, t_chunk, sync=True)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss
