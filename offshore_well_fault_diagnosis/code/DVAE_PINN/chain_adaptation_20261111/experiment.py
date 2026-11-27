"""C-regularization ablation and conditional-response adaptation on 3W 1.1.1."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'adapted_p3_new7_20261110'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 2025]


def load_source():
    original_args = sys.argv
    sys.argv = ['worker', '--version', '1.1.1', '--role', 'preflight']
    try:
        spec = importlib.util.spec_from_file_location('adapted_dvae_source', SOURCE / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = original_args


W = load_source()


def save(path, value):
    W.save(path, value)


def read(path):
    return W.read(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(seed, variant='decay0', starts=None, scales=None):
    W.seed_everything(seed)
    model = W.Model(2, 'full', seed).to(DEVICE)
    state = torch.load(W.OUT / 'runs' / f'seed{seed}_full/initial.pt', map_location=DEVICE)
    model.load_state_dict(state['model'])
    model.with_c = False
    if variant == 'response':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 811)
            model.chain = ResponseChain(starts, scales).to(DEVICE)
    return model


def last_observed(x, mask):
    positions = torch.arange(1, 65, device=x.device).view(1, 64, 1)
    last = (mask[:, :64] * positions).argmax(1)
    anchor = x[:, :64].gather(1, last[:, None, :]).squeeze(1)
    return anchor, mask[:, :64].any(1)


def response_targets(x, mask, starts, scales):
    anchor, observed = last_observed(x, mask)
    targets = torch.stack([x[:, start:start + 16] for start in starts], 1)
    valid = torch.stack([mask[:, start:start + 16] for start in starts], 1)
    valid = valid & observed[:, None, None, :]
    scale = scales.view(1, len(starts), 1, 7)
    return (targets - anchor[:, None, None, :]) / scale, valid, anchor


class ResponseChain(nn.Module):
    """Predict normalized changes at selected horizons using prefix observations."""
    def __init__(self, starts, scales):
        super().__init__()
        self.starts = list(starts)
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float32))
        self.encoder = nn.GRU(22, 64, batch_first=True)
        self.flow_head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(),
                                       nn.Linear(64, len(starts) * 16))
        self.temperature_head = nn.Sequential(nn.Linear(64 + len(starts) * 16, 64),
                                              nn.SiLU(), nn.Linear(64, len(starts) * 32))

    def predict_changes(self, x, mask, prefix):
        history = torch.cat([x[:, :64], mask[:, :64].to(x.dtype), prefix], -1)
        hidden = self.encoder(history)[1][-1]
        flow = self.flow_head(hidden)
        available = (mask[:, :64, 6].float().mean(1) >= .5).to(x.dtype)
        temperature = self.temperature_head(torch.cat([hidden, flow * available[:, None]], -1))
        return (flow.view(-1, len(self.starts), 16, 1),
                temperature.view(-1, len(self.starts), 16, 2))

    def objective(self, x, mask, prefix):
        flow, temperature = self.predict_changes(x, mask, prefix)
        target, valid, _ = response_targets(x, mask, self.starts, self.scales)
        t_loss = W.masked_average(F.smooth_l1_loss(temperature, target[..., [2, 4]], reduction='none'),
                                  valid[..., [2, 4]])
        q_valid = valid[..., [6]] & (mask[:, :64, 6].float().mean(1) >= .5)[:, None, None, None]
        q_loss = W.masked_average(F.smooth_l1_loss(flow, target[..., [6]], reduction='none'), q_valid)
        return t_loss + q_loss

    def temperature_prediction(self, x, mask, prefix):
        _, changes = self.predict_changes(x, mask, prefix)
        _, _, anchor = response_targets(x, mask, self.starts, self.scales)
        prediction = anchor[:, None, None, [2, 4]] + changes * self.scales[:, [2, 4]][None, :, None, :]
        return prediction


def chain_loss(model, x, mask, prefix):
    if isinstance(model.chain, ResponseChain):
        return model.chain.objective(x, mask, prefix)
    return model.chain(x, mask, prefix)[0]


def load_data():
    return W.load_data()


def sample_ids(y, count, seed):
    rng = np.random.default_rng(seed)
    labels = y.cpu().numpy()
    ids = []
    for category in range(9):
        pool = np.flatnonzero(labels == category)
        size = min(len(pool), max(1, round(count * len(pool) / len(labels))))
        ids.extend(rng.choice(pool, size, replace=False))
    return np.array(ids, dtype=np.int64)


@torch.no_grad()
def cache_prefix(model, data, ids):
    model.eval()
    return torch.cat([model.encode(data['x'][batch, :64], data['m'][batch, :64])[0][0].loc
                      for batch in np.array_split(ids, math.ceil(len(ids) / 256))])


def fit_scales(data, ids, starts):
    square = torch.zeros(len(starts), 7, device=DEVICE)
    count = torch.zeros_like(square)
    for batch in np.array_split(ids, math.ceil(len(ids) / 256)):
        delta, valid, _ = response_targets(data['x'][batch], data['m'][batch], starts,
                                          torch.ones(len(starts), 7, device=DEVICE))
        square += (delta.square() * valid).sum((0, 2))
        count += valid.sum((0, 2))
    return (square / count.clamp_min(1)).sqrt().clamp_min(1e-4).cpu().numpy()


@torch.no_grad()
def forecast_metrics(model, data, ids, cached=None):
    model.eval()
    total = baseline = 0.
    valid_count = 0
    latent_change = 0.
    starts = model.chain.starts if isinstance(model.chain, ResponseChain) else [68]
    for begin in range(0, len(ids), 256):
        batch = ids[begin:begin + 256]
        x, mask = data['x'][batch], data['m'][batch]
        prefix = cached[begin:begin + len(batch)] if cached is not None else model.encode(x[:, :64], mask[:, :64])[0][0].loc
        if isinstance(model.chain, ResponseChain):
            prediction = model.chain.temperature_prediction(x, mask, prefix)
            shuffled = model.chain.temperature_prediction(x, mask, prefix.flip(0))
            _, valid_all, anchor = response_targets(x, mask, starts, model.chain.scales)
            valid = valid_all[..., [2, 4]]
            persistence = anchor[:, None, None, [2, 4]]
        else:
            prediction = model.chain.predictions(x, mask, prefix, teacher_forcing=0.)[1][:, None]
            shuffled = model.chain.predictions(x, mask, prefix.flip(0), teacher_forcing=0.)[1][:, None]
            valid = mask[:, 68:84, [2, 4]][:, None]
            persistence = x[:, 63:64, [2, 4]][:, None]
        target = torch.stack([x[:, start:start + 16, [2, 4]] for start in starts], 1)
        total += float(((prediction - target).square() * valid).sum())
        baseline += float(((persistence - target).square() * valid).sum())
        valid_count += int(valid.sum())
        latent_change = max(latent_change, float((prediction - shuffled).abs().max()))
    return {'temperature_mse': total / max(valid_count, 1),
            'persistence_mse': baseline / max(valid_count, 1),
            'persistence_improvement': 1 - total / max(baseline, 1e-30),
            'latent_shuffle_max_prediction_change': latent_change,
            'chain_parameter_norm': float(sum(p.detach().double().square().sum()
                                               for p in model.chain.parameters()).sqrt()),
            'validation_windows': len(ids), 'starts': starts}


def metrics(model, data):
    result, logits = W.evaluate(model, data)
    result['macro_p'] = statistics.mean(value['precision'] for value in result['per_class'].values())
    result['macro_r'] = result['balanced_accuracy']
    return result, logits


def calibrate(model, data, seed):
    model.train()
    model.classifier.eval()
    rng = np.random.default_rng(seed + 1901)
    y = data['y'].cpu().numpy()
    parameters = [p for name, p in model.dvae.named_parameters()
                  if name.startswith(('inference.', 'x_encoder.', 'y_encoder.'))]
    values = []
    for _ in range(4):
        ids = np.concatenate([rng.choice(np.flatnonzero(y == c), 8, False) for c in range(9)])
        x, mask, evidence, labels = [data[key][ids] for key in ['x', 'm', 'e', 'y']]
        ce = 2 * F.cross_entropy(model.predict(x, mask, evidence), labels, weight=model.class_weights)
        main_gradients = torch.autograd.grad(ce, parameters, allow_unused=True)
        prefix = model.encode(x[:, :64], mask[:, :64])[0][0].loc
        auxiliary = chain_loss(model, x, mask, prefix)
        auxiliary_gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        main_norm = sum(g.double().square().sum() for g in main_gradients if g is not None).sqrt()
        aux_norm = sum(g.double().square().sum() for g in auxiliary_gradients if g is not None).sqrt()
        if aux_norm > 0:
            values.append(float(.01 * main_norm / aux_norm))
    if not values:
        raise RuntimeError('Response predictor has zero encoder gradient after warmup')
    weight = float(np.clip(statistics.median(values), 1e-4, .1))
    return {'weight': weight, 'raw_estimates': values, 'target_gradient_fraction': .01,
            'bounds': [1e-4, .1], 'fit_on': 'four balanced training batches'}


def warm_response(seed, starts, data):
    folder = ROOT / 'warmup' / f"seed{seed}_h{'_'.join(map(str, starts))}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    ids = sample_ids(data['train']['y'], 32000, 20261111)
    validation_ids = np.linspace(0, len(data['val']['y']) - 1, 8192, dtype=int)
    scales = fit_scales(data['train'], ids, starts)
    model = make_model(seed, 'response', starts, scales)
    train_prefix = cache_prefix(model, data['train'], ids)
    validation_prefix = cache_prefix(model, data['val'], validation_ids)
    optimizer = torch.optim.Adam(model.chain.parameters(), lr=.001, weight_decay=0.)
    best = float('inf')
    history = []
    begin = 0
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location=DEVICE)
        model.chain.load_state_dict(checkpoint['chain'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        W.restore_rng(checkpoint['rng'])
        best, history, begin = checkpoint['best'], checkpoint['history'], checkpoint['next_epoch']
    for epoch in range(begin, 3):
        model.chain.train()
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(ids))
        for start in range(0, len(order), 256):
            positions = order[start:start + 256]
            batch = ids[positions]
            loss = model.chain.objective(data['train']['x'][batch], data['train']['m'][batch],
                                         train_prefix[positions])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.chain.parameters(), 5.)
            optimizer.step()
        report = forecast_metrics(model, data['val'], validation_ids, validation_prefix)
        history.append({'epoch': epoch + 1, **report})
        score = report['temperature_mse'] / max(report['persistence_mse'], 1e-30)
        if score < best:
            best = score
            W.atomic_torch_save(folder / 'best.pt', {'chain': model.chain.state_dict(),
                                                     'starts': starts, 'scales': scales,
                                                     'forecast': report})
        W.atomic_torch_save(folder / 'last.pt', {'chain': model.chain.state_dict(),
                                               'optimizer': optimizer.state_dict(), 'rng': W.rng_state(),
                                               'next_epoch': epoch + 1, 'best': best, 'history': history})
        save(folder / 'progress.json', history[-1])
    result = {'seed': seed, 'starts': starts, 'scales': scales.tolist(),
              'best_forecast_ratio': best, 'history': history, 'test_read': False}
    save(folder / 'result.json', result)
    return result


def train(seed, variant, data, max_epochs=10):
    folder = ROOT / 'runs' / f'seed{seed}_{variant}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return
    response = variant == 'response'
    selection = read(ROOT / 'response_selection.json') if response else None
    starts = selection['starts'] if response else None
    warm = ROOT / 'warmup' / f"seed{seed}_h{'_'.join(map(str, starts))}" if response else None
    warmed = torch.load(warm / 'best.pt', map_location=DEVICE) if response else None
    model = make_model(seed, variant, starts, warmed['scales'] if response else None)
    if response:
        model.chain.load_state_dict(warmed['chain'])
    model.with_c = False
    primary = [p for name, p in model.named_parameters() if not name.startswith('chain.')]
    optimizer = torch.optim.Adam([
        {'params': primary, 'lr': .0003, 'weight_decay': 1e-5},
        {'params': model.chain.parameters(), 'lr': .001 if response else .0003, 'weight_decay': 0.},
    ])
    calibration = calibrate(model, data['train'], seed) if response else {'weight': .001}
    save(folder / 'calibration.json', calibration)
    history, begin, best, stale = [], 0, -1., 0
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        W.restore_rng(checkpoint['rng'])
        begin, history = checkpoint['next_epoch'], checkpoint['history']
        best, stale, calibration = checkpoint['best'], checkpoint['stale'], checkpoint['calibration']
    else:
        original = torch.load(W.OUT / 'runs' / f'seed{seed}_rule/initial.pt', map_location=DEVICE)['model']
        assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items()
                   if not name.startswith('chain.'))
        W.atomic_torch_save(folder / 'initial.pt', {'model': model.state_dict(), 'calibration': calibration})
    n_validation = len(data['val']['y'])
    validation_ids = np.linspace(0, n_validation - 1, min(n_validation, 8192), dtype=int)
    for epoch in range(begin, max_epochs):
        model.train()
        order = np.random.default_rng(seed * 10000 + epoch).permutation(len(data['train']['y']))
        tic = time.monotonic()
        coefficient = calibration['weight'] * (min(1., (epoch + 1) / 3) if response else 1.)
        for step, start in enumerate(range(0, len(order), 32)):
            ids = order[start:start + 32]
            torch.manual_seed(seed * 1000000 + epoch * 10000 + step)
            batch = data['train']
            x, mask, y, evidence = [batch[key][ids] for key in ['x', 'm', 'y', 'e']]
            primary_loss, _ = model.loss(x, mask, y, evidence)
            prefix = model.encode(x[:, :64], mask[:, :64])[0][0].loc
            auxiliary = chain_loss(model, x, mask, prefix)
            loss = primary_loss + coefficient * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {seed}/{variant}/{epoch}/{step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if step % 1000 == 0:
                save(folder / 'progress.json', {'epoch': epoch + 1, 'step': step,
                                                'steps': math.ceil(len(order) / 32), 'best': best})
        validation, logits = metrics(model, data['val'])
        forecast = forecast_metrics(model, data['val'], validation_ids)
        record = {'epoch': epoch + 1, 'validation_p': validation['macro_p'],
                  'validation_r': validation['macro_r'], 'validation_f1': validation['macro_f1'],
                  'C_weight': coefficient, 'forecast': forecast, 'seconds': time.monotonic() - tic}
        history.append(record)
        if validation['macro_f1'] > best:
            best, stale = validation['macro_f1'], 0
            W.atomic_torch_save(folder / 'best.pt', {'model': model.state_dict(), 'epoch': epoch + 1,
                                                     'metrics': validation, 'variant': variant,
                                                     'starts': starts, 'scales': warmed['scales'] if response else None})
            np.savez_compressed(folder / 'validation_predictions.npz', logits=logits)
        else:
            stale += 1
        W.atomic_torch_save(folder / 'last.pt', {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': W.rng_state(),
            'next_epoch': epoch + 1, 'best': best, 'stale': stale, 'history': history,
            'calibration': calibration,
        })
        save(folder / 'progress.json', record)
        print(seed, variant, record, flush=True)
        if stale >= 4:
            break
    save(folder / 'result.json', {'seed': seed, 'variant': variant, 'best_val_f1': best,
                                 'history': history, 'epochs': len(history), 'test_read': False})


def preflight():
    train_data = {}
    for part in ['train']:
        ids = np.arange(72)
        train_data = {
            'x': torch.tensor(np.load(W.BASE / f'cache/{part}/x.npy', mmap_mode='r')[ids], device=DEVICE),
            'm': torch.tensor(np.load(W.BASE / f'cache/{part}/mask.npy', mmap_mode='r')[ids], device=DEVICE, dtype=torch.bool),
            'e': torch.tensor(np.load(SOURCE / f'1.1.1/cache/{part}_e.npy', mmap_mode='r')[ids], device=DEVICE),
            'y': torch.tensor(np.load(W.BASE / f'cache/{part}/index.npy')[ids, 2].astype(np.int64), device=DEVICE),
        }
    x, mask, y, evidence = [train_data[key] for key in ['x', 'm', 'y', 'e']]
    model = make_model(42)
    original = W.Model(2, 'full', 42).to(DEVICE)
    original.load_state_dict(model.state_dict())
    model.eval()
    original.eval()
    assert torch.equal(model.predict(x, mask, evidence), original.predict(x, mask, evidence))
    torch.manual_seed(100)
    first = model.loss(x, mask, y, evidence)[0]
    prefix = model.encode(x[:, :64], mask[:, :64])[0][0].loc
    first = first + .001 * chain_loss(model, x, mask, prefix)
    torch.manual_seed(100)
    second = original.loss(x, mask, y, evidence)[0]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
    for starts in [[68], [112], [68, 112]]:
        response = make_model(42, 'response', starts, np.ones((len(starts), 7)))
        response.train()
        z = response.encode(x[:, :64], mask[:, :64])[0][0].loc
        objective = chain_loss(response, x, mask, z)
        objective.backward()
        assert torch.isfinite(objective)
        assert all(torch.isfinite(p.grad).all() for p in response.parameters() if p.grad is not None)
        response.eval()
        with torch.no_grad():
            before = response.chain.temperature_prediction(x, mask, z.detach())
            changed = x.clone()
            changed[:, 64:] += 100
            after = response.chain.temperature_prediction(changed, mask, z.detach())
            assert torch.equal(before, after)
    save(ROOT / 'preflight.json', {'status': 'passed', 'original_R_C_objective_matches': True,
                                   'main_logits_identical': True, 'response_finite_backward': True,
                                   'future_inputs_do_not_change_prediction': True})


def test_and_report():
    assert all((ROOT / 'runs' / f'seed{seed}_{variant}/result.json').exists()
               for seed in SEEDS for variant in ['decay0', 'response'])
    rows = []
    x = np.load(W.BASE / 'cache/test/x.npy', mmap_mode='r')
    masks = np.load(W.BASE / 'cache/test/mask.npy', mmap_mode='r')
    evidence = np.load(SOURCE / '1.1.1/cache/test_e.npy', mmap_mode='r')
    index = np.load(W.BASE / 'cache/test/index.npy')
    for seed in SEEDS:
        for variant in ['decay0', 'response']:
            folder = ROOT / 'runs' / f'seed{seed}_{variant}'
            destination = folder / 'test_result.json'
            if destination.exists():
                rows.append(read(destination))
                continue
            checkpoint = torch.load(folder / 'best.pt', map_location=DEVICE)
            model = make_model(seed, variant, checkpoint['starts'], checkpoint['scales'])
            model.load_state_dict(checkpoint['model'])
            model.eval()
            logits = []
            with torch.no_grad():
                for begin in range(0, len(index), 256):
                    end = begin + 256
                    logits.append(model.predict(torch.tensor(np.array(x[begin:end]), device=DEVICE),
                                                torch.tensor(np.array(masks[begin:end]), device=DEVICE, dtype=torch.bool),
                                                torch.tensor(np.array(evidence[begin:end]), device=DEVICE)).cpu().numpy())
            logits = np.concatenate(logits)
            assert np.isfinite(logits).all()
            report = W.classification_metrics(index[:, 2], logits.argmax(1))
            report['macro_p'] = statistics.mean(v['precision'] for v in report['per_class'].values())
            report['macro_r'] = report['balanced_accuracy']
            report.update(seed=seed, variant=variant, selected_epoch=checkpoint['epoch'])
            np.savez_compressed(folder / 'test_predictions.npz', logits=logits, labels=index[:, 2], sources=index[:, 3])
            save(destination, report)
            rows.append(report)
    for seed in SEEDS:
        for role, variant in [('rule', 'R'), ('full', 'old_RC')]:
            report = read(SOURCE / '1.1.1/runs' / f'seed{seed}_{role}/test_result.json')
            rows.append({**report, 'variant': variant})
    lines = ['# 1.1.1 chain loss适配结果', '', '种子42和2025，测试集均值±样本标准差。', '',
             '| 配置 | Macro-P | Macro-R | Macro-F1 |', '|---|---:|---:|---:|']
    for variant, name in [('R', '适配DVAE＋新7D＋R'), ('old_RC', '＋原C'),
                          ('decay0', '＋C独立零衰减'), ('response', '＋归一化变化预测C与预热/渐入')]:
        values = [r for r in rows if r['variant'] == variant]
        summary = []
        for key in ['macro_p', 'macro_r', 'macro_f1']:
            summary.append(f"{statistics.mean(r[key] for r in values):.6f} ± {statistics.stdev(r[key] for r in values):.6f}")
        lines.append('| ' + name + ' | ' + ' | '.join(summary) + ' |')
    lines += ['', '响应预测方案额外进行三轮、约32000训练窗口的预测头预热。联合阶段预算采用最多10轮和patience4，参数由训练/验证数据确定。']
    (ROOT / 'final_results.md').write_text('\n'.join(lines), encoding='utf-8')
    save(ROOT / 'test_rows.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['preflight', 'train', 'warm', 'test'], required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--variant', choices=['decay0', 'response'], default='decay0')
    parser.add_argument('--starts', default='68,112')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight()
    elif args.action == 'test':
        test_and_report()
    else:
        data = load_data()
        if args.action == 'warm':
            warm_response(args.seed, list(map(int, args.starts.split(','))), data)
        else:
            train(args.seed, args.variant, data)


if __name__ == '__main__':
    main()
