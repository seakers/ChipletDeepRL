import numpy as np
import torch
from torch import nn

from utils.construct import EMPTY, stack, place_clear, partial_costs, random_structure
from utils.cost_counter import COUNTS
from optimization.construct_search import grid_outputs


def mlp(n_in, width=64):
    net = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    # zero output layer: an untrained policy is uniform (or exactly the prior)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


N_STRUCT_FEAT = 3 + 3 + 1 + 1 + 5
N_PLACE_FEAT = 4 + 4 + 3 + 2 + 4 + 1 + 5


def structure_features(s, w):
    shape = np.eye(3)[int(s[0])]
    dims = np.array(s[1:4]) / 2.0
    return np.concatenate([shape, dims, [np.prod(s[1:4]) / 8.0, s[4] / 3.0], w])


def placement_features(ev, x, i, cost, before, scale, step, w):
    comp = ev.component_list[i]
    sdims = np.maximum(np.abs(np.array(x[1:4], dtype=float)), 1e-3)
    loc = np.asarray(comp.location, dtype=float) / sdims
    on_shelf = float(x[5 + 5 * i] >= 8)
    comp_feat = [comp.mass / 10.0] + list(np.asarray(comp.dimensions, dtype=float))
    return np.concatenate([cost / scale, (cost - before) / scale, loc, [on_shelf, float(comp.pointing)],
                           comp_feat, [step], w])


def logits_of(net, feats, prior):
    return net(torch.as_tensor(np.array(feats), dtype=torch.float32)).squeeze(-1) + torch.as_tensor(prior)


def choose(rng, net, feats, prior):
    with torch.no_grad():
        probs = torch.softmax(logits_of(net, feats, prior), 0).numpy().astype(float)
    k = int(rng.choice(len(probs), p=probs / probs.sum()))
    return k, float(np.log(probs[k] + 1e-12))


def build_design(ev, rng, struct_net, place_net, w, scale, m=4, S=8, prior_b=0.0):
    """One build. Returns (design or None, decisions); decisions are (net, feats, prior, choice, logp)."""
    COUNTS['constructions'] += 1
    COUNTS['structure_attempts'] += 1
    cands = [random_structure(ev, rng) for _ in range(S)]
    feats = [structure_features(s, w) for s in cands]
    no_prior = np.zeros(S, dtype=np.float32)
    k, lp = choose(rng, struct_net, feats, no_prior)
    steps = [(0, feats, no_prior, k, lp)]
    w4 = w[:4] / w[:4].sum()
    x = list(cands[k]) + [0] * (5 * ev.num_components)
    ev.get_structural_panels(x)
    pcu = [j for j, c in enumerate(ev.component_list) if c.type == 'PCU']
    order = pcu + [int(j) for j in rng.permutation(ev.num_components) if j not in pcu]
    placed, done = EMPTY, []
    before = np.zeros(4)
    for n, i in enumerate(order):
        options = []
        for _ in range(m):
            new = place_clear(ev, x, i, placed, rng, 500)
            if new is None:
                break
            cost = partial_costs(ev, done + [i])
            f = placement_features(ev, x, i, cost, before, scale, n / len(order), w)
            options.append((f, list(x[5 + 5 * i:10 + 5 * i]), new, cost))
        if not options:
            return None, steps
        score = np.array([w4 @ (o[3] / scale) for o in options])
        prior = (-prior_b * (score - score.mean()) / (score.std() + 1e-12)).astype(np.float32)
        k, lp = choose(rng, place_net, [o[0] for o in options], prior)
        steps.append((1, [o[0] for o in options], prior, k, lp))
        x[5 + 5 * i:10 + 5 * i] = options[k][1]
        ev.place_component(i, x)
        placed = stack(placed, options[k][2])
        done.append(i)
        before = options[k][3]
    return x, steps


def sample_designs(ev, rng, path, n, max_obj, prior_b=3.0, m=4, S=8):
    """n designs from a saved policy (construct_ppo.pth), each with random objective weights."""
    struct_net, place_net = mlp(N_STRUCT_FEAT), mlp(N_PLACE_FEAT)
    state = torch.load(path)
    struct_net.load_state_dict(state['struct'])
    place_net.load_state_dict(state['place'])
    scale = np.asarray(max_obj, dtype=float)[:4]
    out = []
    while len(out) < n:
        x, _ = build_design(ev, rng, struct_net, place_net, rng.dirichlet(np.ones(5)), scale, m, S, prior_b)
        if x is not None:
            out.append(x)
    return out


def run_construct_ppo(max_obj, params, eval_function):
    """
    Constructive PPO: builds each design like construct_greedy, but a policy picks the structure
    (out of S random ones) and each component's placement (out of m clear candidates), so every
    design it evaluates is feasible. The policy is conditioned on random objective weights; the
    16 designs of a batch share the weights and the reward is the negative weighted objective sum,
    standardized within the batch. Structures that cannot be filled get a fixed penalty without an
    evaluation. With cppo_prior = b > 0 the placement logits start from construct_greedy's rule
    (b times the standardized weighted partial cost), so the frozen policy is close to greedy and
    the network learns a correction.
    """
    ev = eval_function
    seed = params.get('seed')
    rng = np.random.default_rng(seed)
    torch.manual_seed(0 if seed is None else seed)
    torch.set_num_threads(1)
    budget = params.get('nfe_budget') or params['num_epochs'] * params['mini_batch_size']
    m = params.get('cppo_m', 4)
    S = params.get('cppo_structures', 8)
    batch = params.get('cppo_batch', 16)
    lr = params.get('cppo_lr', 3e-4)
    clip = params.get('cppo_clip', 0.2)
    epochs = params.get('cppo_epochs', 4)
    fail_adv = params.get('cppo_fail_adv', 1.0)  # advantage of a failed build, in batch std units
    prior_b = params.get('cppo_prior', 0.0)
    max_obj = np.asarray(max_obj, dtype=float)
    scale = max_obj[:4]

    struct_net, place_net = mlp(N_STRUCT_FEAT), mlp(N_PLACE_FEAT)
    opt = torch.optim.Adam(list(struct_net.parameters()) + list(place_net.parameters()), lr=lr)

    def update(episodes):
        # finished designs are ranked against each other; failed builds get a fixed penalty
        r = np.array([e[1] for e in episodes if e[1] is not None])
        if params.get('cppo_frozen') or len(r) < 2 or r.std() == 0:
            return
        adv = [(e[1] - r.mean()) / r.std() if e[1] is not None else -fail_adv for e in episodes]
        data = [(net, feats, prior, k, lp, a) for (steps, _), a in zip(episodes, adv)
                for (net, feats, prior, k, lp) in steps]
        for _ in range(epochs):
            loss = 0.0
            for net_id, feats, prior, k, lp_old, a in data:
                net = struct_net if net_id == 0 else place_net
                logp = torch.log_softmax(logits_of(net, feats, prior), 0)[k]
                ratio = torch.exp(logp - lp_old)
                loss = loss - torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
            opt.zero_grad()
            (loss / len(data)).backward()
            opt.step()

    all_des, all_obj, all_constraints = [], [], []
    fails = 0
    while len(all_obj) < budget:
        w = rng.dirichlet(np.ones(5))
        episodes = []
        while len(episodes) < batch and len(all_obj) < budget:
            x, steps = build_design(ev, rng, struct_net, place_net, w, scale, m, S, prior_b)
            if x is None:
                # unfillable structure: penalize the choices that led here, no evaluation
                episodes.append((steps, None))
                fails += 1
                continue
            objectives, violated, _ = ev.evaluate(x)
            all_des.append(x)
            all_obj.append(objectives)
            all_constraints.append(violated)
            norm = np.asarray(objectives, dtype=float) / max_obj
            episodes.append((steps, -float(w @ norm) if not violated else None))
        update(episodes)

    print(f'[cppo] {fails} unfillable structures in {len(all_obj)} designs')
    if params.get('save_path'):
        torch.save({'struct': struct_net.state_dict(), 'place': place_net.state_dict()},
                   params['save_path'] + 'construct_ppo.pth')
    return grid_outputs(all_des, all_obj, all_constraints, max_obj, ev.num_objectives)
