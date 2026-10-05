import time
from contextlib import nullcontext

import torch
from torch import autocast


# params['legacy_ppo'] = True turns off all the PPO fixes listed in the README
def is_legacy(params):
    return bool(params.get('legacy_ppo', False))


def amp_settings(params):
    # dropout and fp16 make the PPO ratio != 1 at the first update, so both default off
    legacy = is_legacy(params)
    return params.get('use_amp', legacy), params.get('dropout', 0.1 if legacy else 0.0)


def amp_ctx(module):
    if getattr(module, 'use_amp', False):
        return autocast(device_type=module.device.type, dtype=torch.float16)
    return nullcontext()


def add_positional_encoding(pos_enc, x, legacy=False):
    # PositionalEncoding expects (seq, batch, d) but x is (batch, seq, d)
    if legacy:
        return pos_enc(x)
    return pos_enc(x.transpose(0, 1)).transpose(0, 1)


def gather_positions(out, pos):
    # out[b, pos[b]], or the last position if pos is None
    if pos is None:
        return out[:, -1]
    return out[torch.arange(out.size(0), device=out.device), pos.long()]


def unique_var_index(var_idx, num_unique=10, num_struct=5):
    # unique_des_space is the 5 structure vars followed by one component's 5 vars
    if var_idx < num_struct:
        return var_idx
    return num_struct + (var_idx - num_struct) % (num_unique - num_struct)


def throttle(params, t_start):
    # params['gpu_duty'] < 1 sleeps after each epoch so the GPU is busy only that fraction of the time
    duty = params.get('gpu_duty', 1.0)
    if duty < 1.0:
        time.sleep((time.time() - t_start) * (1.0 / duty - 1.0))
