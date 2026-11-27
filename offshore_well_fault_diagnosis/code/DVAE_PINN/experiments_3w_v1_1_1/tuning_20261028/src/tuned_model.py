"""Configurable 1.1.1 adaptation; the prior experiment remains immutable."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from tuning_common import BASE, ROOT
from models import Diagnoser, masked_average
from torch_dvae.DVAE import DVAE
from torch_dvae.encoder_models import RNNEncoder
from torch_dvae.inference_models import RNNInference
from torch_dvae.initializers import ControlInitializer
from torch_dvae.measurement_models import MLPMeasure
from torch_dvae.transition_models import MLPTransition


class TunedDiagnoser(Diagnoser):
    def __init__(self, features, cfg):
        super().__init__(features, cfg.get("evidence", False), cfg.get("chain", False), cfg.get("rule", False), cfg.get("zdim", 8), cfg.get("hidden", 64))
        self.cfg = cfg
        self.use_mask = cfg.get("mask_input", False)
        self.pool = cfg.get("pool", "mean")
        zdim, hidden = cfg.get("zdim", 8), cfg.get("hidden", 64)
        if self.use_mask:
            obs = len(features)
            d = obs*2
            xe, ye = RNNEncoder(d, hidden), RNNEncoder(d, hidden)
            self.dvae = DVAE(RNNInference(zdim, hidden, hidden, hidden), MLPTransition(zdim, hidden, 0, hidden), MLPMeasure(obs, zdim, hidden, hidden, hidden), ye, xe, ControlInitializer(hidden, zdim, 0, d))
        if self.pool == "stats":
            self.classifier = nn.Sequential(nn.Linear(zdim*3 + (7 if self.use_evidence else 0), hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, 9))
        self.register_buffer("class_weights", torch.ones(9))

    def encode_masked(self, x, mask):
        inp = torch.cat([x, mask.to(x.dtype)], -1) if self.use_mask else x
        return self.dvae.inference_func(inp, inp), inp

    def logits(self, z):
        evidence = self.evidence(z) if self.use_evidence else None
        h = z.mean(1)
        if self.pool == "stats":
            h = torch.cat([h, z.std(1, unbiased=False), z[:, -1]-z[:, 0]], -1)
        if evidence is not None:
            h = torch.cat([h, evidence.tanh()], -1)
        return self.classifier(h), evidence

    def predict(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        posterior = self.encode_masked(x, mask)[0][0]
        return self.logits(posterior.loc)[0]

    def loss(self, x, mask, y, epoch=0):
        (posterior_raw, yhist, _, _, xencoded), inp = self.encode_masked(x, mask)
        posterior = Normal(posterior_raw.loc, posterior_raw.scale.clamp(.02, 5))
        logits, evidence = self.logits(posterior.loc)
        ce = F.cross_entropy(logits, y, weight=self.class_weights, label_smoothing=self.cfg.get("label_smoothing", 0.))
        rec = ce.new_zeros(())
        kl = ce.new_zeros(())
        rec_weight = self.cfg.get("rec_weight", 1.)
        beta = self.cfg.get("beta", .001)
        if rec_weight or beta:
            z = posterior.rsample() if self.cfg.get("sampling", "rsample") == "rsample" else posterior.sample()
            init_raw = self.dvae.init_net(inp, inp)
            init = Normal(init_raw.loc, init_raw.scale.clamp(.02, 5)).rsample()
            prior_raw = self.dvae.get_priors(init, z, yhist, xencoded)
            prior = Normal(prior_raw.loc, prior_raw.scale.clamp(.02, 5))
            reconstructed = self.dvae.measure.get_dist(z, yhist, xencoded)
            mode = self.cfg.get("rec_mode", "fixed")
            if mode == "fixed":
                nll = .5*(reconstructed.loc-x).square() + .5*math.log(2*math.pi)
            else:
                sigma = reconstructed.scale.clamp(self.cfg.get("sigma_floor", .1), 5)
                nll = -Normal(reconstructed.loc, sigma).log_prob(x)
            rec = masked_average(nll, mask)
            # Legacy control retains its original T scaling and unbounded
            # observation variance to separate effects of objective scaling.
            if mode == "legacy":
                rec = masked_average(-reconstructed.log_prob(x), mask)*x.shape[1]
            kl = kl_divergence(posterior, prior).mean()
        warmup = min(1., (epoch+1)/max(1, self.cfg.get("kl_warmup", 1)))
        rule = F.cross_entropy(self.evidence.rule_head(evidence.tanh()), y, weight=self.class_weights) if self.use_rule else ce.new_zeros(())
        chain = ce.new_zeros(())
        if self.use_chain:
            prefix_z = self.encode_masked(x[:, :64], mask[:, :64])[0][0].loc
            chain, _ = self.chain(x, mask, prefix_z)
        total = rec_weight*rec + beta*warmup*kl + self.cfg.get("cls_weight", 2.)*ce + self.cfg.get("rule_weight", .0005)*rule + self.cfg.get("chain_weight", .001)*chain
        return total, {"loss": total.detach(), "rec": rec.detach(), "kl": kl.detach(), "ce": ce.detach(), "rule": rule.detach(), "chain": chain.detach()}

    def load_initial(self, state):
        """Allow only documented additive input/head extensions."""
        own = self.state_dict()
        for name, value in state.items():
            if name == "class_weights" or name not in own:
                continue
            if own[name].shape == value.shape:
                own[name] = value.clone()
            elif name == "classifier.0.weight" and own[name].shape[0] == value.shape[0] and own[name].shape[1] >= value.shape[1]:
                own[name].zero_(); own[name][:, :value.shape[1]] = value
            elif name.endswith("rnn_fwd.weight_ih_l0") and own[name].shape[1] == value.shape[1]*2:
                own[name].zero_(); own[name][:, :value.shape[1]] = value
            elif name.endswith("rnn_bwd.weight_ih_l0") and self.use_mask:
                # Vendor input order: [forward hidden, observations, mask].
                own[name].zero_(); own[name][:, :value.shape[1]] = value
            elif name == "dvae.init_net.net.net.linear1.weight" and self.use_mask:
                own[name].zero_(); own[name][:, :value.shape[1]] = value
            else:
                raise ValueError(f"Unsupported checkpoint shape change: {name}: {value.shape} -> {own[name].shape}")
        self.load_state_dict(own)
