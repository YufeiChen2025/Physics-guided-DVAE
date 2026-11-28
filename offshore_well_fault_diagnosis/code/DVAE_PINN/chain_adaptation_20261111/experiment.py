"""Train the 3W 1.1.1 models with the zero-decay conditional predictor."""
import importlib.util
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'adapted_p3_new7_20261110'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_source():
    spec = importlib.util.spec_from_file_location('adapted_dvae_source', SOURCE / 'worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

W = load_source()


def save(path, value):
    W.save(path, value)


def make_model(seed):
    W.seed_everything(seed)
    model = W.Model(2, 'full', seed).to(DEVICE)
    state = torch.load(W.OUT / 'runs' / f'seed{seed}_full/initial.pt', map_location=DEVICE)
    model.load_state_dict(state['model'])
    model.with_c = False
    return model


def chain_loss(model, x, mask, prefix):
    return model.chain(x, mask, prefix)[0]


@torch.no_grad()
def forecast_metrics(model, data, ids):
    model.eval()
    total = baseline = 0.
    valid_count = 0
    latent_change = 0.
    starts = [68]
    for begin in range(0, len(ids), 256):
        batch = ids[begin:begin + 256]
        x, mask = data['x'][batch], data['m'][batch]
        prefix = model.encode(x[:, :64], mask[:, :64])[0][0].loc
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


def train(seed, data, max_epochs=10):
    variant = 'decay0'
    folder = ROOT / 'runs' / f'seed{seed}_{variant}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return
    model = make_model(seed)
    model.with_c = False
    primary = [p for name, p in model.named_parameters() if not name.startswith('chain.')]
    optimizer = torch.optim.Adam([
        {'params': primary, 'lr': .0003, 'weight_decay': 1e-5},
        {'params': model.chain.parameters(), 'lr': .0003, 'weight_decay': 0.},
    ])
    calibration = {'weight': .001}
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
        if model.with_e:
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
        coefficient = calibration['weight']
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
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
                                                     'metrics': validation, 'variant': variant})
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
