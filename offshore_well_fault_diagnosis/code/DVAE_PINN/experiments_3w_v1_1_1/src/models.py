"""Nine-class DVAE, latent-only 7D proxies, and observation-masked auxiliary chain.

This is a version-specific adaptation, not a claim that latent proxies are
calibrated physical measurements. The original 2.0 implementation is untouched.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.distributions as distributions
from torch import nn
from torch.nn import functional as F

from common import CODE

sys.path.insert(0, str(CODE / "vendor"))
from torch_dvae.DVAE import DVAE
from torch_dvae.encoder_models import RNNEncoder
from torch_dvae.inference_models import RNNInference
from torch_dvae.initializers import ControlInitializer
from torch_dvae.measurement_models import MLPMeasure
from torch_dvae.transition_models import MLPTransition


def masked_average(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class LatentEvidence7D(nn.Module):
    names = ["water_rise", "water_smooth", "dhsv_open_mean", "pck_drop", "pck_open_mean", "dhsv_low_evidence", "pck_low_evidence"]

    def __init__(self, zdim=8):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(zdim, zdim), nn.Tanh(), nn.Linear(zdim, 1))
        self.water_cut_head = head()
        self.dhsv_opening_head = head()
        self.pck_opening_head = head()
        self.rule_head = nn.Linear(7, 9)
        # The retained seven columns of the existing engineering prior;
        # no indexing of class 9, and no fabricated observed valve states.
        w = torch.zeros(9, 7)
        w[0, 1] = -.05
        w[0, 2] = .1; w[0, 4] = .1
        w[0, 5] = -.1; w[0, 6] = -.1
        w[1, 0] = .2; w[1, 1] = .05
        w[2, 2] = -.25; w[2, 5] = 1.
        w[5, 2] = .15; w[5, 4] = .15
        w[5, 5] = -.2; w[5, 6] = -.2
        w[6, 3] = 1.; w[6, 6] = 1.
        w[7, 3] = .125; w[7, 6] = .5
        w[8, 4] = .15
        with torch.no_grad():
            self.rule_head.weight.copy_(w)
            self.rule_head.bias.zero_()

    def forward(self, z):
        water = self.water_cut_head(z).sigmoid().squeeze(-1)
        dhsv = self.dhsv_opening_head(z).sigmoid().squeeze(-1)
        pck = self.pck_opening_head(z).sigmoid().squeeze(-1)
        q = max(1, z.shape[1] // 4)
        features = [water[:, -q:].mean(1) - water[:, :q].mean(1),
                    (water[:, 1:] - water[:, :-1]).square().mean(1).add(1e-12).sqrt(),
                    dhsv.mean(1), pck[:, :q].mean(1) - pck[:, -q:].mean(1),
                    pck.mean(1), F.relu(.85 - dhsv).mean(1), F.relu(.35 - pck).mean(1)]
        return torch.stack(features, -1)


class MaskedChain(nn.Module):
    """P->QGL->T where QGL history exists; P/T fallback otherwise.

    Only prefix-encoded latent states are accepted, so future targets cannot
    leak through a bidirectional encoder into the auxiliary predictors.
    """
    history, lag, horizon = 64, 4, 16

    def __init__(self, features, zdim=8, hidden=64):
        super().__init__()
        self.p = [i for i, name in enumerate(features) if name in ["P-PDG", "P-TPT", "P-MON-CKP", "P-JUS-CKGL"]]
        self.t = [i for i, name in enumerate(features) if name in ["T-TPT", "T-JUS-CKP"]]
        self.q = [features.index("QGL")] if "QGL" in features else []
        if not self.p or not self.t:
            raise ValueError("Pressure and temperature channels are required")
        self.pressure_encoder = nn.GRU(len(self.p) + zdim, hidden, batch_first=True)
        self.temp_encoder = nn.GRU(len(self.t) + zdim, hidden, batch_first=True)
        self.flow_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(.05), nn.Linear(hidden, self.horizon))
        self.flow_encoder = nn.GRU(1, hidden, batch_first=True)
        self.flow_gate = nn.Sequential(nn.Linear(hidden*3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.temp_head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Dropout(.05), nn.Linear(hidden, self.horizon*len(self.t)))
        self.log_sigma_q = nn.Parameter(torch.full((self.horizon, 1), -1.))
        self.log_sigma_t = nn.Parameter(torch.full((self.horizon, len(self.t)), -2.))

    def predictions(self, x, mask, prefix_z, teacher_forcing=0.):
        ph, th = x[:, :self.history, self.p], x[:, :self.history, self.t]
        if prefix_z.shape[1] != self.history:
            raise ValueError("Chain requires latent encoded from the 64-point prefix only")
        hp = self.pressure_encoder(torch.cat([ph, prefix_z], -1))[1][-1]
        ht = self.temp_encoder(torch.cat([th, prefix_z], -1))[1][-1]
        q_mu = self.flow_head(hp).view(-1, self.horizon, 1)
        q_used = q_mu
        begin = self.history + self.lag
        if self.q:
            available = mask[:, :self.history, self.q].any(1).to(x.dtype)
            if self.training and teacher_forcing:
                qmask = mask[:, begin:begin+self.horizon, self.q].to(x.dtype)
                target = x[:, begin:begin+self.horizon, self.q]
                q_used = q_mu + teacher_forcing * qmask * (target - q_mu)
        else:
            available = x.new_zeros(x.shape[0], 1)
        hq = self.flow_encoder(q_used)[1][-1]
        state = ht + self.flow_gate(torch.cat([hp, ht, hq], -1)) * hq * available
        residual = self.temp_head(torch.cat([state, hp], -1)).view(-1, self.horizon, len(self.t))
        t_mu = x[:, self.history-1:self.history, self.t] + .1 * residual
        return q_mu, t_mu, available

    def forward(self, x, mask, prefix_z, teacher_forcing=.15):
        q_mu, t_mu, available = self.predictions(x, mask, prefix_z, teacher_forcing)
        start = self.history + self.lag
        target_t = x[:, start:start+self.horizon, self.t]
        mask_t = mask[:, start:start+self.horizon, self.t]
        def nll(target, mu, log_sigma, valid):
            logs = log_sigma.clamp(-6, 3)
            losses = .5 * ((target-mu).square() * torch.exp(-2*logs) + 2*logs + math.log(2*math.pi))
            return masked_average(losses, valid)
        t_loss = nll(target_t, t_mu, self.log_sigma_t, mask_t)
        q_loss = q_mu.sum() * 0
        q_count = 0
        if self.q:
            target_q = x[:, start:start+self.horizon, self.q]
            mask_q = mask[:, start:start+self.horizon, self.q]
            q_loss = nll(target_q, q_mu, self.log_sigma_q, mask_q)
            q_count = int(mask_q.sum().detach())
        return .005*q_loss + t_loss, {"chain_q_nll": q_loss.detach(), "chain_t_nll": t_loss.detach(), "chain_q_targets": q_count, "chain_t_targets": int(mask_t.sum().detach()), "chain_q_history_fraction": available.mean().detach()}


class Diagnoser(nn.Module):
    def __init__(self, features, evidence=False, chain=False, rule=False, zdim=8, hidden=64):
        super().__init__()
        self.features, self.use_evidence, self.use_chain, self.use_rule = list(features), evidence, chain, rule
        if rule and not evidence:
            raise ValueError("Rule loss requires evidence features")
        d = len(features)
        xe, ye = RNNEncoder(d, hidden), RNNEncoder(d, hidden)
        self.dvae = DVAE(RNNInference(zdim, hidden, hidden, hidden), MLPTransition(zdim, hidden, 0, hidden), MLPMeasure(d, zdim, hidden, hidden, hidden), ye, xe, ControlInitializer(hidden, zdim, 0, d))
        self.classifier = nn.Sequential(nn.Linear(zdim + (7 if evidence else 0), hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, 9))
        self.evidence = LatentEvidence7D(zdim) if evidence else None
        self.chain = MaskedChain(features, zdim, hidden) if chain else None

    def encode(self, x):
        return self.dvae.inference_func(x, x)

    def logits(self, z):
        evidence = self.evidence(z) if self.use_evidence else None
        h = z.mean(1)
        if evidence is not None:
            h = torch.cat([h, evidence.tanh()], -1)
        return self.classifier(h), evidence

    def predict(self, x):
        q = self.encode(x)[0]
        return self.logits(q.loc)[0]

    def loss(self, x, mask, y):
        posterior, yhist, _, _, xencoded = self.encode(x)
        # Reparameterized sample: reconstruction gradients reach the encoder.
        # Applied identically to every group; documented deviation from the old
        # 2.0 training script's .sample(), not a direct 2.0 score reproduction.
        z = posterior.rsample()
        z0 = self.dvae.init_net(x, x).rsample()
        prior = self.dvae.get_priors(z0, z, yhist, xencoded)
        reconstructed = self.dvae.measure.get_dist(z, yhist, xencoded)
        nll_per_point = -reconstructed.log_prob(x)
        per_window = (nll_per_point*mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
        rec = per_window.mean() * x.shape[1]
        kl = distributions.kl_divergence(posterior, prior).sum(1).mean()
        logits, evidence = self.logits(posterior.loc)
        ce = F.cross_entropy(logits, y)
        rule = F.cross_entropy(self.evidence.rule_head(evidence.tanh()), y) if self.use_rule else ce.new_zeros(())
        chain = ce.new_zeros(())
        parts = {}
        if self.use_chain:
            prefix_z = self.encode(x[:, :64])[0].loc
            chain, parts = self.chain(x, mask, prefix_z)
        loss = rec + .001*kl + 2.*ce + .0005*rule + .001*chain
        parts.update({"loss": loss.detach(), "reconstruction": rec.detach(), "kl": kl.detach(), "ce": ce.detach(), "rule_ce": rule.detach(), "chain": chain.detach()})
        return loss, parts

    def warm_start(self, state):
        """Preserve baseline logits; new evidence weights start at exactly zero."""
        own = self.state_dict()
        copied, extended = [], []
        for key, value in state.items():
            if key not in own:
                continue
            if own[key].shape == value.shape:
                own[key] = value.clone()
                copied.append(key)
            elif key == "classifier.0.weight" and own[key].shape[0] == value.shape[0] and own[key].shape[1] == value.shape[1]+7:
                own[key].zero_()
                own[key][:, :value.shape[1]] = value
                extended.append(key)
            else:
                raise ValueError(f"Unexpected warm-start shape mismatch: {key}")
        if not copied or any(k.startswith("dvae.") and k not in copied for k in own):
            raise ValueError("Incomplete DVAE warm-start")
        self.load_state_dict(own)
        return {"copied_tensors": len(copied), "extended": extended}
