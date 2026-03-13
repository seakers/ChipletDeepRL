import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer - same as truss version [5]"""
    
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
    """
    Design repair actor for spacecraft configuration.
    Takes complete design as input, outputs:
    1. Variable to modify (pointer network)
    2. New value for that variable (continuous or discrete based on variable type)
    3. Stop indicator
    
    Combines repair mechanism from [5] with spacecraft design space handling from [3].
    """
    
    def __init__(self, device, params, des_space, comp_list, num_objectives, repair_mode=True):
        super(Actor, self).__init__()
        self.device = device
        self.params = params
        self.des_space = des_space  # Unique design space definition
        self.comp_list = comp_list  # Component list for panel validity
        self.num_objectives = num_objectives
        self.repair_mode = repair_mode
        
        self.num_variables = len(des_space)
        self.dense_dim = params.get('dense_dim', 16)
        self.nhead = 2
        self.num_layers = 2
        self.clip_ratio = params['clip_ratio']
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input dimension: design variables + objectives
        self.input_dim = self.num_variables + num_objectives
        
        # Encode input design
        self.design_encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        
        # Transformer encoder to process design
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dense_dim,
            nhead=self.nhead,
            dim_feedforward=self.dense_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Pointer network for variable selection (attention-based) [5]
        self.pointer_query = nn.Linear(self.dense_dim, self.dense_dim)
        self.pointer_key = nn.Linear(self.dense_dim, self.dense_dim)
        
        # Value prediction heads - one per unique variable type [3]
        # For spacecraft: mix of continuous and discrete variables
        self.value_predictors = nn.ModuleList()
        for var in des_space:
            if var['type'] == 'continuous':
                # Beta distribution parameters (alpha, beta) [3]
                self.value_predictors.append(nn.Sequential(
                    nn.Linear(self.dense_dim * 2, self.dense_dim),
                    nn.ReLU(),
                    nn.Linear(self.dense_dim, 2)  # alpha, beta for Beta distribution
                ))
            elif var['type'] == 'discrete':
                # Categorical distribution over discrete options
                self.value_predictors.append(nn.Sequential(
                    nn.Linear(self.dense_dim * 2, self.dense_dim),
                    nn.ReLU(),
                    nn.Linear(self.dense_dim, len(var['range']))
                ))
            else:
                raise ValueError(f"Invalid design space type: {var['type']}")
        
        # Stop decision head [5]
        self.stop_head = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim // 2),
            nn.ReLU(),
            nn.Linear(self.dense_dim // 2, 2)  # [continue, stop]
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    
    def forward(self, design_state, var_idx=None):
        """
        design_state: [batch, num_variables + num_objectives] or [num_variables + num_objectives]
        Returns pointer logits, encoded features, and stop logits
        """
        if design_state.dim() == 1:
            design_state = design_state.unsqueeze(0)
            
        with autocast(device_type=self.device.type, dtype=torch.float16):
            batch_size = design_state.size(0)
            
            # Encode each element of the design separately
            design_encoded = self.design_encoder(design_state.unsqueeze(-1))  # [batch, seq_len, dense_dim]
            design_encoded = self.positional_encoding(design_encoded.transpose(0, 1)).transpose(0, 1)
            
            # Transform through encoder
            encoded = self.transformer_encoder(design_encoded)  # [batch, seq_len, dense_dim]
            
            # Pointer network: which variable to modify?
            context = encoded.mean(dim=1, keepdim=True)  # [batch, 1, dense_dim]
            query = self.pointer_query(context)  # [batch, 1, dense_dim]
            keys = self.pointer_key(encoded[:, :self.num_variables, :])  # Only point to design variables, not objectives
            
            # Attention scores for variable selection
            pointer_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)  # [batch, num_variables]
            pointer_logits = pointer_logits / math.sqrt(self.dense_dim)
            
            # Stop decision (based on context)
            stop_logits = self.stop_head(context.squeeze(1))  # [batch, 2]
            
            return pointer_logits, encoded, stop_logits
    
    
    def get_valid_panel_mask(self, design_state, var_idx):
        """
        Get valid panel mask based on structure_id and shelves.
        Adapted from [3] and [4] for the spacecraft configuration problem.
        design_state: [batch, seq_len] - contains design variables
        var_idx: which variable index we're selecting a value for
        """
        batch_size = design_state.size(0)
        num_panel_options = len(self.des_space[5]['range']) if len(self.des_space) > 5 else 14

        # Extract structure_id (index 0) and shelves (index 4) from design
        structure_ids = design_state[:, 0].long()
        shelves = design_state[:, 4].long() if design_state.size(1) > 4 else torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Base panels per structure type [4]
        base_panels_lookup = torch.tensor([5, 6, 8], device=self.device)
        base_panels = base_panels_lookup[torch.clamp(structure_ids, 0, 2)]

        mask = torch.zeros((batch_size, num_panel_options), dtype=torch.bool, device=self.device)

        # Determine if this is a pointing component
        # Panel choice indices are at index 5 or every 5th index after that [3][4]
        comp_idx = (var_idx - 5) // 5 if var_idx >= 5 else 0
        is_pointing = self.comp_list[comp_idx].pointing if comp_idx < len(self.comp_list) else True

        for b in range(batch_size):
            bp = base_panels[b].item()
            sh = shelves[b].item()

            # Base panels are always valid
            mask[b, 0:bp] = True

            # Non-pointing components can also use shelf panels [3][4]
            if not is_pointing:
                mask[b, 8:8+sh] = True   # Shelf front sides
                mask[b, 11:11+sh] = True  # Shelf back sides

        return mask


    def get_value_logits(self, encoded, var_indices, design_state=None):
        """
        Given encoded design and selected variable indices, predict new values.
        Handles both continuous and discrete variables [3].
        encoded: [batch, seq_len, dense_dim]
        var_indices: [batch] - which variable to modify
        design_state: [batch, seq_len] - original design (needed for panel mask)
        """
        batch_size = encoded.size(0)

        with autocast(device_type=self.device.type, dtype=torch.float16):
            # Gather the encoded representation of selected variables
            var_indices_expanded = var_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.dense_dim)
            selected_var_encoding = torch.gather(encoded, 1, var_indices_expanded).squeeze(1)  # [batch, dense_dim]

            # Context from whole design
            context = encoded.mean(dim=1)  # [batch, dense_dim]

            # Concatenate for value prediction
            combined = torch.cat([selected_var_encoding, context], dim=-1)  # [batch, dense_dim * 2]

            # Get value logits for each unique variable type
            # We need to handle the case where different items in batch may have different var_indices
            all_value_outputs = []
            all_var_types = []

            for i in range(batch_size):
                var_idx = var_indices[i].item()

                # Map to unique design space index (handle repeated components)
                unique_idx = var_idx % len(self.des_space) if var_idx >= len(self.des_space) else var_idx

                value_output = self.value_predictors[unique_idx](combined[i:i+1])
                all_value_outputs.append(value_output)
                all_var_types.append(self.des_space[unique_idx]['type'])

            return all_value_outputs, all_var_types, var_indices


    def sample_action(self, design_state, full_des_space):
        """
        Sample variable to change, new value, and stop decision for repair.
        Handles mixed continuous/discrete design space [3].
        Returns: (var_log_prob, var_idx, value_log_prob, value_idx, stop_log_prob, stop_decision)
        """
        if design_state.dim() == 1:
            design_state = design_state.unsqueeze(0)

        pointer_logits, encoded, stop_logits = self.forward(design_state)

        # FIX: Convert to float32 before creating distribution
        pointer_logits_fp32 = pointer_logits.float()
        var_dist = torch.distributions.Categorical(logits=pointer_logits_fp32)
        var_indices = var_dist.sample()  # [batch]
        var_log_probs = var_dist.log_prob(var_indices)  # [batch]

        # Sample new value for selected variable
        value_outputs, var_types, _ = self.get_value_logits(encoded, var_indices, design_state)

        value_indices_list = []
        value_log_probs_list = []

        for i, (value_output, var_type) in enumerate(zip(value_outputs, var_types)):
            var_idx = var_indices[i].item()
            unique_idx = var_idx % len(self.des_space) if var_idx >= len(self.des_space) else var_idx

            if var_type == 'continuous':
                # FIX: Convert to float32 before Beta distribution
                params = torch.exp(value_output.float())  # Ensure positive and fp32
                alpha = params[0, 0] + 1e-6
                beta = params[0, 1] + 1e-6
                value_dist = torch.distributions.Beta(alpha, beta)
                value_sample = value_dist.sample()
                value_log_prob = value_dist.log_prob(value_sample)
                value_indices_list.append(value_sample)
                value_log_probs_list.append(value_log_prob)

            elif var_type == 'discrete':
                # FIX: Convert to float32 BEFORE applying mask
                logits = value_output.squeeze(0).float()

                # Apply panel validity mask if this is a panel choice variable [3][4]
                if unique_idx == 5 or (unique_idx > 5 and (unique_idx - 5) % 5 == 0):
                    valid_mask = self.get_valid_panel_mask(design_state[i:i+1], var_idx)
                    # FIX: Now safe to use -1e9 since logits is float32
                    logits[~valid_mask.squeeze(0)] = -1e9

                # FIX: Use logits parameter instead of probs for numerical stability
                value_dist = torch.distributions.Categorical(logits=logits)
                value_sample = value_dist.sample()
                value_log_prob = value_dist.log_prob(value_sample)
                value_indices_list.append(value_sample.float())
                value_log_probs_list.append(value_log_prob)

        value_indices = torch.stack(value_indices_list)
        value_log_probs = torch.stack(value_log_probs_list)

        # FIX: Convert to float32 before creating distribution
        stop_logits_fp32 = stop_logits.float()
        stop_dist = torch.distributions.Categorical(logits=stop_logits_fp32)
        stop_decisions = stop_dist.sample()  # [batch], 0=continue, 1=stop
        stop_log_probs = stop_dist.log_prob(stop_decisions)

        return (var_log_probs.squeeze(), var_indices.squeeze(),
                value_log_probs.squeeze(), value_indices.squeeze(),
                stop_log_probs.squeeze(), stop_decisions.squeeze())


    def ppo_update(self, observations, actions, logprobs, advantages):
        """
        PPO update for all three outputs.
        Handles mixed continuous/discrete variables [3][5].
        """
        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type, dtype=torch.float16):
            policy_losses = []
            all_kls = []

            for idx in range(len(observations)):
                observation = observations[idx]
                action = actions[idx]  # [timesteps, 3] - var_idx, value_idx, stop
                logprob = logprobs[idx]  # [timesteps, 3]
                advantage = advantages[idx]

                if observation.size(0) == 0:
                    continue

                # Forward pass
                pointer_logits, encoded, stop_logits = self.forward(observation)

                # FIX: Convert to float32 before log_softmax for numerical stability
                var_log_probs = F.log_softmax(pointer_logits.float(), dim=-1)
                var_new_log_probs = var_log_probs.gather(1, action[:, 0].unsqueeze(-1).long()).squeeze(-1)

                # Recompute log probs for value selection
                value_new_log_probs = []

                for t in range(observation.size(0)):
                    var_idx = action[t, 0].long().item()
                    unique_idx = var_idx % len(self.des_space) if var_idx >= len(self.des_space) else var_idx
                    var_type = self.des_space[unique_idx]['type']

                    # Get value prediction for this timestep
                    var_indices_t = action[t:t+1, 0].long()
                    value_outputs_t, _, _ = self.get_value_logits(
                        encoded[t:t+1], var_indices_t, observation[t:t+1]
                    )
                    value_output = value_outputs_t[0]

                    if var_type == 'continuous':
                        # FIX: Convert to float32 before Beta distribution
                        params = torch.exp(value_output.float())
                        alpha = params[0, 0] + 1e-6
                        beta = params[0, 1] + 1e-6
                        value_dist = torch.distributions.Beta(alpha, beta)
                        # Clamp action to valid Beta range
                        action_val = torch.clamp(action[t, 1], 1e-6, 1-1e-6)
                        val_log_prob = value_dist.log_prob(action_val)
                    else:
                        # FIX: Convert to float32 BEFORE applying mask
                        logits = value_output.squeeze(0).float()

                        if unique_idx == 5 or (unique_idx > 5 and (unique_idx - 5) % 5 == 0):
                            valid_mask = self.get_valid_panel_mask(observation[t:t+1], var_idx)
                            # FIX: Now safe to use -1e9 since logits is float32
                            logits[~valid_mask.squeeze(0)] = -1e9

                        # FIX: Use float32 for log_softmax
                        val_log_probs = F.log_softmax(logits, dim=-1)
                        val_log_prob = val_log_probs[action[t, 1].long()]

                    value_new_log_probs.append(val_log_prob)

                value_new_log_probs = torch.stack(value_new_log_probs)

                # FIX: Convert to float32 before log_softmax
                stop_log_probs = F.log_softmax(stop_logits.float(), dim=-1)
                stop_new_log_probs = stop_log_probs.gather(1, action[:, 2].unsqueeze(-1).long()).squeeze(-1)

                # Combined log prob (joint probability)
                old_log_probs = torch.sum(logprob, dim=-1)
                new_log_probs = var_new_log_probs + value_new_log_probs + stop_new_log_probs

                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                min_advantage = torch.where(
                    advantage > 0,
                    (1 + self.clip_ratio) * advantage,
                    (1 - self.clip_ratio) * advantage
                )
                policy_loss = -torch.min(ratio * advantage, min_advantage).mean()
                policy_losses.append(policy_loss)

                # FIX: Track KL for each batch item
                all_kls.append(torch.mean(new_log_probs - old_log_probs))

            # FIX: Stack losses instead of cat (they're scalars)
            policy_loss = torch.stack(policy_losses).mean()

        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # FIX: Compute KL from tracked values
        kl = torch.stack(all_kls).mean()

        return policy_loss.item(), kl.item()


class Critic(nn.Module):
    """
    Value network for spacecraft design repair.
    Estimates expected return from current design state.
    Supports weighted multi-objective evaluation similar to [3].
    """
    
    def __init__(self, device, params, num_objectives, input_dim):
        super(Critic, self).__init__()
        self.device = device
        self.params = params
        self.num_objectives = num_objectives
        self.input_dim = input_dim
        
        self.dense_dim = params.get('dense_dim', 16)
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input layer to match spacecraft design space dimensions [3]
        self.input_layer = nn.Linear(self.input_dim, self.dense_dim)
        self.hidden_layer = nn.Linear(self.dense_dim, self.dense_dim)
        # Output per objective for weighted aggregation [3]
        self.output_layer = nn.Linear(self.dense_dim, self.num_objectives)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    
    def forward(self, x):
        """
        x: [batch, input_dim] - design variables + objectives
        Returns: value estimates [batch, num_objectives]
        """
        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = F.relu(self.input_layer(x))
            x = F.relu(self.hidden_layer(x))
            x = self.output_layer(x)
        return x
    
    
    def sample_critic(self, observation):
        """
        Evaluate value for batch of design states.
        Handles variable-length observations by padding [3].
        
        observation: list of observations or tensor
        Returns: value estimates [batch, num_objectives]
        """
        if isinstance(observation, list):
            input_observations = []
            for obs in observation:
                input_obs = list(obs) if not isinstance(obs, list) else obs.copy()
                # Pad to input_dim if needed [3]
                while len(input_obs) < self.input_dim:
                    input_obs.append(0)
                input_observations.append(input_obs[:self.input_dim])
            observation = torch.tensor(input_observations, dtype=torch.float32, device=self.device)
        elif not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        if observation.device != self.device:
            observation = observation.to(self.device)
            
        return self.forward(observation)
    
    
    def ppo_update(self, design_states, returns, weights=None):
        """
        Update critic with PPO.
        Supports weighted multi-objective returns [3].
        
        design_states: list of tensors [timesteps, input_dim]
        returns: list of tensors [timesteps]
        weights: optional weights for multi-objective aggregation [3]
        """
        self.optimizer.zero_grad()
        value_losses = []
        
        with autocast(device_type=self.device.type, dtype=torch.float16):
            for idx in range(len(design_states)):
                if design_states[idx].size(0) == 0:
                    continue
                    
                predicted_values = self.forward(design_states[idx])  # [timesteps, num_objectives]
                
                if weights is not None and len(weights) > idx:
                    # Weighted aggregation of predicted values [3]
                    batch_weights = weights[idx]
                    if not isinstance(batch_weights, torch.Tensor):
                        batch_weights = torch.tensor(batch_weights, dtype=torch.float32, device=self.device)
                    
                    # Expand weights if needed
                    if batch_weights.dim() == 1:
                        batch_weights = batch_weights.unsqueeze(0).expand(predicted_values.size(0), -1)
                    
                    predicted_reward = torch.sum(-predicted_values * batch_weights, dim=-1)
                else:
                    # Simple mean across objectives if no weights provided
                    predicted_reward = predicted_values.mean(dim=-1)
                
                # MSE loss against returns [5]
                value_loss = (predicted_reward - returns[idx]) ** 2
                value_losses.append(value_loss)
            
            if len(value_losses) == 0:
                return 0.0
                
            value_loss = torch.cat(value_losses).mean()
        
        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return value_loss.item()