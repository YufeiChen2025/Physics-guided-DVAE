from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from threew.dataset import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    ThreeWWindowDataset,
    compute_normalization_stats,
    list_parquet_files,
    stratified_split_by_instance,
)
from torch_dvae.DVAE import DVAE
from torch_dvae.encoder_models import EmptyEncoder, RNNEncoder
from torch_dvae.inference_models import RNNInference
from torch_dvae.initializers import ControlInitializer
from torch_dvae.measurement_models import MLPMeasure
from torch_dvae.transition_models import MLPTransition
from torch_dvae.utils import configure_paper_architecture


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LatentClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_seq_mean: torch.Tensor) -> torch.Tensor:
        return self.net(z_seq_mean)


class ChainPriorAuxLoss(nn.Module):
    """Latent-conditioned P->Q->T auxiliary prior for DVAE training.

    The loss uses observed normalized windows as supervision, but conditions the
    chain predictors on DVAE posterior latent history. This lets the auxiliary
    task shape the main DVAE representation instead of training a detached pilot.
    """

    def __init__(
        self,
        feature_columns: list[str],
        zdim: int,
        pressure_cols: list[str],
        flow_cols: list[str],
        temp_cols: list[str],
        history_len: int,
        lag: int,
        pred_len: int,
        hidden_dim: int,
        residual_scale: float,
        dropout: float,
        use_physical_a: bool = False,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.col = {name: i for i, name in enumerate(self.feature_columns)}
        self.pressure_idx = self._indices(pressure_cols)
        self.flow_idx = self._indices(flow_cols)
        self.temp_idx = self._indices(temp_cols)
        self.history_len = history_len
        self.lag = lag
        self.pred_len = pred_len
        self.residual_scale = residual_scale
        self.use_physical_a = use_physical_a
        self.flow_cols = list(flow_cols)
        self.temp_cols = list(temp_cols)

        pressure_dim = len(self.pressure_idx) + zdim
        temp_dim = len(self.temp_idx) + zdim
        flow_dim = len(self.flow_idx)
        temp_out_dim = len(self.temp_idx)

        self.pressure_encoder = nn.GRU(pressure_dim, hidden_dim, batch_first=True)
        self.temp_encoder = nn.GRU(temp_dim, hidden_dim, batch_first=True)
        self.flow_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * flow_dim),
        )
        self.flow_future_encoder = nn.GRU(flow_dim, hidden_dim, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.physical_a_encoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.physical_a_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.temp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * temp_out_dim),
        )
        self.log_sigma_q = nn.Parameter(torch.full((pred_len, flow_dim), -1.0))
        self.log_sigma_t = nn.Parameter(torch.full((pred_len, temp_out_dim), -2.0))

    def _indices(self, cols: list[str]) -> list[int]:
        missing = [name for name in cols if name not in self.col]
        if missing:
            raise KeyError(f"Missing 3W feature columns for chain prior: {missing}")
        return [self.col[name] for name in cols]

    @staticmethod
    def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-2.0 * log_sigma)
        return 0.5 * (((target - mu) ** 2) * inv_var + 2.0 * log_sigma + np.log(2.0 * np.pi)).mean()

    def forward(
        self,
        x_norm: torch.Tensor,
        z_mu: torch.Tensor,
        flow_weight: float,
        temp_weight: float,
        teacher_forcing: float,
        physical_future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_norm.shape[1] < self.history_len + self.lag + self.pred_len:
            raise ValueError("DVAE window is shorter than history_len + lag + pred_len.")

        target_start = self.history_len + self.lag
        pressure_history = x_norm[:, : self.history_len, self.pressure_idx]
        temp_history = x_norm[:, : self.history_len, self.temp_idx]
        latent_history = z_mu[:, : self.history_len]
        q_target = x_norm[:, target_start : target_start + self.pred_len, self.flow_idx]
        t_target = x_norm[:, target_start : target_start + self.pred_len, self.temp_idx]
        temp_anchor = x_norm[:, self.history_len - 1 : self.history_len, self.temp_idx].expand_as(t_target)

        pressure_input = torch.cat([pressure_history, latent_history], dim=-1)
        temp_input = torch.cat([temp_history, latent_history], dim=-1)

        _p_seq, p_hidden = self.pressure_encoder(pressure_input)
        _t_seq, t_hidden = self.temp_encoder(temp_input)
        hp = p_hidden[-1]
        ht = t_hidden[-1]

        q_mu = self.flow_head(hp).view(x_norm.shape[0], self.pred_len, len(self.flow_idx))
        if self.training and teacher_forcing > 0.0:
            q_for_temp = teacher_forcing * q_target + (1.0 - teacher_forcing) * q_mu
        else:
            q_for_temp = q_mu

        _q_seq, q_hidden = self.flow_future_encoder(q_for_temp)
        hq = q_hidden[-1]
        flow_gate = self.gate(torch.cat([hp, ht, hq], dim=-1))
        chain_state = ht + flow_gate * hq
        if self.use_physical_a and physical_future is not None:
            if physical_future.ndim == 2:
                physical_future = physical_future.unsqueeze(-1)
            if physical_future.shape[1] != self.pred_len:
                raise ValueError("physical_future length must match chain pred_len.")
            _a_seq, a_hidden = self.physical_a_encoder(physical_future)
            ha = a_hidden[-1]
            a_gate = self.physical_a_gate(torch.cat([hp, ht, hq, ha], dim=-1))
            chain_state = chain_state + a_gate * ha
        t_residual = self.temp_head(torch.cat([chain_state, hp], dim=-1)).view(
            x_norm.shape[0], self.pred_len, len(self.temp_idx)
        )
        t_mu = temp_anchor + self.residual_scale * t_residual

        q_log_sigma = self.log_sigma_q.unsqueeze(0).expand_as(q_mu).clamp(min=-6.0, max=3.0)
        t_log_sigma = self.log_sigma_t.unsqueeze(0).expand_as(t_mu).clamp(min=-6.0, max=3.0)
        q_nll = self.gaussian_nll(q_target, q_mu, q_log_sigma)
        t_nll = self.gaussian_nll(t_target, t_mu, t_log_sigma)
        q_mse = ((q_target - q_mu) ** 2).mean()
        t_mse = ((t_target - t_mu) ** 2).mean()
        loss = flow_weight * q_nll + temp_weight * t_nll
        parts = {
            "chain_total": loss.detach(),
            "chain_flow_nll": q_nll.detach(),
            "chain_temp_nll": t_nll.detach(),
            "chain_flow_mse": q_mse.detach(),
            "chain_temp_mse": t_mse.detach(),
        }
        if self.use_physical_a and physical_future is not None:
            parts["chain_physical_a_mean"] = physical_future.detach().mean()
            parts["chain_physical_a_drop"] = (
                physical_future.detach()[:, 0].mean() - physical_future.detach()[:, -1].mean()
            )
        return loss, parts


class LatentPhysicalAuxLoss(nn.Module):
    """Expose an interpretable opening/restriction variable from z_mu.

    The head reads DVAE posterior latents and produces one bounded physical
    scalar per time step. It uses weak supervision from an observed valve column
    when available, plus class-conditional trend constraints.
    """

    def __init__(
        self,
        feature_columns: list[str],
        stats: dict[str, np.ndarray],
        zdim: int,
        observed_col: str,
        fault_label: int,
        normal_label: int,
        hidden_dim: int,
        mode: str,
        trend_margin: float,
        sequence_head: bool = False,
        latent_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"decrease", "increase"}:
            raise ValueError("mode must be 'decrease' or 'increase'.")
        self.feature_columns = list(feature_columns)
        self.col = {name: i for i, name in enumerate(self.feature_columns)}
        if observed_col not in self.col:
            raise KeyError(f"Missing observed physical column: {observed_col}")
        self.observed_col = observed_col
        self.observed_idx = self.col[observed_col]
        self.fault_label = fault_label
        self.normal_label = normal_label
        self.mode = mode
        self.trend_margin = trend_margin
        self.sequence_head = sequence_head
        if latent_indices is None:
            latent_indices = list(range(zdim))
        latent_indices = [idx + zdim if idx < 0 else idx for idx in latent_indices]
        if not latent_indices or min(latent_indices) < 0 or max(latent_indices) >= zdim:
            raise ValueError(f"physical latent indices must be within [0, {zdim - 1}], got {latent_indices}")
        self.latent_indices = list(latent_indices)
        complement = [idx for idx in range(zdim) if idx not in set(self.latent_indices)]
        self.register_buffer("latent_index_tensor", torch.as_tensor(self.latent_indices, dtype=torch.long), persistent=False)
        self.register_buffer("state_index_tensor", torch.as_tensor(complement, dtype=torch.long), persistent=False)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        input_zdim = len(self.latent_indices)
        self.head = nn.Sequential(
            nn.Linear(input_zdim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.sequence_net = nn.GRU(input_zdim, hidden_dim, batch_first=True)
        self.sequence_head_out = nn.Linear(hidden_dim, 1)

    def _restriction_latent(self, z_mu: torch.Tensor) -> torch.Tensor:
        return z_mu.index_select(-1, self.latent_index_tensor.to(device=z_mu.device))

    def latent_regularizers(self, z_mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_restriction = self._restriction_latent(z_mu)
        l2_loss = z_restriction.pow(2).mean()
        if self.state_index_tensor.numel() == 0:
            return z_restriction.mean() * 0.0, l2_loss
        z_state = z_mu.index_select(-1, self.state_index_tensor.to(device=z_mu.device))
        zr = z_restriction.reshape(-1, z_restriction.shape[-1])
        zs = z_state.reshape(-1, z_state.shape[-1])
        zr = (zr - zr.mean(dim=0, keepdim=True)) / zr.std(dim=0, keepdim=True).clamp_min(1e-6)
        zs = (zs - zs.mean(dim=0, keepdim=True)) / zs.std(dim=0, keepdim=True).clamp_min(1e-6)
        cov = zr.transpose(0, 1) @ zs / max(zr.shape[0] - 1, 1)
        return cov.pow(2).mean(), l2_loss

    def opening(self, z_mu: torch.Tensor) -> torch.Tensor:
        z_phys = self._restriction_latent(z_mu)
        if self.sequence_head:
            seq, _hidden = self.sequence_net(z_phys)
            return torch.sigmoid(self.sequence_head_out(seq)).squeeze(-1)
        return torch.sigmoid(self.head(z_phys)).squeeze(-1)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=values.dtype, device=values.device)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _quarter_means(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = max(values.shape[1] // 4, 1)
        return values[:, :q].mean(dim=1), values[:, -q:].mean(dim=1)

    def forward(
        self,
        x_norm: torch.Tensor,
        z_mu: torch.Tensor,
        align_weight: float,
        trend_weight: float,
        normal_weight: float,
        smooth_weight: float,
        profile_weight: float,
        profile_start: float,
        profile_end: float,
        normal_flat_weight: float = 0.0,
        decorrelation_weight: float = 0.0,
        restriction_l2_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_hat = self.opening(z_mu)
        x_units = x_norm * self.std.view(1, 1, -1) + self.mean.view(1, 1, -1)
        observed = x_units[..., self.observed_idx]
        obs_finite = torch.isfinite(observed)
        obs_target = torch.clamp(observed, 0.0, 1.0)
        align_loss = self._masked_mean((a_hat - obs_target) ** 2, obs_finite)

        first, last = self._quarter_means(a_hat)
        if self.mode == "decrease":
            trend_amount = first - last
        else:
            trend_amount = last - first
        fault_mask = getattr(self, "_current_fault_mask", None)
        normal_mask = getattr(self, "_current_normal_mask", None)
        if fault_mask is not None and bool(fault_mask.any()):
            trend_loss = self._masked_mean(F.relu(self.trend_margin - trend_amount), fault_mask)
        else:
            trend_loss = a_hat.mean() * 0.0
        if normal_mask is not None:
            normal_loss = self._masked_mean(F.relu(0.85 - a_hat).mean(dim=1), normal_mask)
            normal_flat_loss = self._masked_mean((first - last).abs(), normal_mask)
        else:
            normal_loss = a_hat.mean() * 0.0
            normal_flat_loss = a_hat.mean() * 0.0

        if a_hat.shape[1] > 1:
            da = a_hat[:, 1:] - a_hat[:, :-1]
            smooth_loss = da.abs().mean()
        else:
            smooth_loss = a_hat.mean() * 0.0
        if fault_mask is not None and bool(fault_mask.any()):
            if self.mode == "decrease":
                target_line = torch.linspace(
                    profile_start, profile_end, a_hat.shape[1], device=a_hat.device, dtype=a_hat.dtype
                )
            else:
                target_line = torch.linspace(
                    profile_end, profile_start, a_hat.shape[1], device=a_hat.device, dtype=a_hat.dtype
                )
            target_line = target_line.view(1, -1).expand_as(a_hat)
            profile_loss = self._masked_mean((a_hat - target_line) ** 2, fault_mask)
        else:
            profile_loss = a_hat.mean() * 0.0
        decorrelation_loss, restriction_l2_loss = self.latent_regularizers(z_mu)

        total = (
            align_weight * align_loss
            + trend_weight * trend_loss
            + normal_weight * normal_loss
            + smooth_weight * smooth_loss
            + profile_weight * profile_loss
            + normal_flat_weight * normal_flat_loss
            + decorrelation_weight * decorrelation_loss
            + restriction_l2_weight * restriction_l2_loss
        )
        parts = {
            "phys_total": total.detach(),
            "phys_align": align_loss.detach(),
            "phys_trend": trend_loss.detach(),
            "phys_normal": normal_loss.detach(),
            "phys_normal_flat": normal_flat_loss.detach(),
            "phys_smooth": smooth_loss.detach(),
            "phys_profile": profile_loss.detach(),
            "phys_decorrelation": decorrelation_loss.detach(),
            "phys_restriction_l2": restriction_l2_loss.detach(),
            "phys_open_mean": a_hat.detach().mean(),
        }
        return total, parts

    def forward_with_labels(
        self,
        x_norm: torch.Tensor,
        z_mu: torch.Tensor,
        labels: torch.Tensor,
        align_weight: float,
        trend_weight: float,
        normal_weight: float,
        smooth_weight: float,
        profile_weight: float,
        profile_start: float,
        profile_end: float,
        normal_flat_weight: float = 0.0,
        decorrelation_weight: float = 0.0,
        restriction_l2_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._current_fault_mask = labels == self.fault_label
        self._current_normal_mask = labels == self.normal_label
        try:
            return self.forward(
                x_norm,
                z_mu,
                align_weight,
                trend_weight,
                normal_weight,
                smooth_weight,
                profile_weight,
                profile_start,
                profile_end,
                normal_flat_weight,
                decorrelation_weight,
                restriction_l2_weight,
            )
        finally:
            self._current_fault_mask = None
            self._current_normal_mask = None


class StructuredRestrictionTransition(MLPTransition):
    """MLP transition with a physical residual path for selected latent dims."""

    def __init__(
        self,
        zdim: int,
        hdim: int,
        ydim: int,
        xdim: int,
        latent_indices: list[int],
        hidden_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__(zdim, hdim, ydim, xdim)
        latent_indices = [idx + zdim if idx < 0 else idx for idx in latent_indices]
        if not latent_indices or min(latent_indices) < 0 or max(latent_indices) >= zdim:
            raise ValueError(f"transition latent indices must be within [0, {zdim - 1}], got {latent_indices}")
        self.latent_indices = list(latent_indices)
        self.residual_scale = residual_scale
        self.register_buffer("latent_index_tensor", torch.as_tensor(self.latent_indices, dtype=torch.long), persistent=False)
        input_dim = len(self.latent_indices) * 2 + 1
        self.opening_head = nn.Sequential(
            nn.Linear(len(self.latent_indices), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(self.latent_indices)),
        )
        self._last_delta: torch.Tensor | None = None
        self._last_opening: torch.Tensor | None = None

    def _restriction_latent(self, zs: torch.Tensor) -> torch.Tensor:
        return zs.index_select(-1, self.latent_index_tensor.to(device=zs.device))

    def opening_from_z(self, zs: torch.Tensor) -> torch.Tensor:
        z_phys = self._restriction_latent(zs)
        return torch.sigmoid(self.opening_head(z_phys)).squeeze(-1)

    def get_dist(self, zs, ys, xs):
        base_dist = super().get_dist(zs, ys, xs)
        z_phys = self._restriction_latent(zs)
        prev = torch.cat([torch.zeros_like(z_phys[:, :1]), z_phys[:, :-1]], dim=1)
        dz_phys = z_phys - prev
        opening = torch.sigmoid(self.opening_head(z_phys))
        features = torch.cat([z_phys, dz_phys, opening], dim=-1)
        delta = torch.tanh(self.delta_head(features)) * self.residual_scale
        loc = base_dist.loc.clone()
        loc.index_copy_(
            -1,
            self.latent_index_tensor.to(device=loc.device),
            loc.index_select(-1, self.latent_index_tensor.to(device=loc.device)) + delta,
        )
        self._last_delta = delta
        self._last_opening = opening.squeeze(-1)
        return dists.Normal(loc, base_dist.scale)

    def aux_loss(
        self,
        z_mu: torch.Tensor,
        physical_opening: torch.Tensor | None,
        delta_l2_weight: float,
        physical_consistency_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = z_mu.mean() * 0.0
        delta_l2 = self._last_delta.pow(2).mean() if self._last_delta is not None else zero
        if physical_opening is not None:
            transition_opening = self.opening_from_z(z_mu)
            consistency = (transition_opening - physical_opening.detach()).pow(2).mean()
        else:
            transition_opening = self.opening_from_z(z_mu)
            consistency = zero
        total = delta_l2_weight * delta_l2 + physical_consistency_weight * consistency
        parts = {
            "transition_total": total.detach(),
            "transition_delta_l2": delta_l2.detach(),
            "transition_physical_consistency": consistency.detach(),
            "transition_open_mean": transition_opening.detach().mean(),
        }
        return total, parts


class DVAEFaultDiagnoser(nn.Module):
    def __init__(
        self,
        dvae: DVAE,
        classifier: LatentClassifier,
        pinn_loss: nn.Module | None = None,
        chain_prior_aux: ChainPriorAuxLoss | None = None,
        physical_aux: LatentPhysicalAuxLoss | None = None,
        use_rule_evidence_features: bool = False,
        use_rule_energy_logits: bool = False,
        rule_energy_logit_weight: float = 0.0,
    ):
        super().__init__()
        self.dvae = dvae
        self.classifier = classifier
        self.pinn_loss = pinn_loss
        self.chain_prior_aux = chain_prior_aux
        self.physical_aux = physical_aux
        self.use_rule_evidence_features = use_rule_evidence_features
        self.use_rule_energy_logits = use_rule_energy_logits
        self.rule_energy_logit_weight = rule_energy_logit_weight

    def rule_evidence(
        self,
        x_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z_mu: torch.Tensor,
    ) -> torch.Tensor | None:
        if not (self.use_rule_evidence_features or self.use_rule_energy_logits):
            return None
        if self.pinn_loss is None or not hasattr(self.pinn_loss, "evidence_features"):
            raise ValueError("rule evidence/logit fusion requires a pinn loss with evidence_features().")
        return self.pinn_loss.evidence_features(x_norm=x_norm, recon_norm=recon_norm, z_mu=z_mu)

    def classifier_input(
        self,
        x_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z_mu: torch.Tensor,
        evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_summary = z_mu.mean(dim=1)
        if not self.use_rule_evidence_features:
            return z_summary
        if evidence is None:
            evidence = self.rule_evidence(x_norm, recon_norm, z_mu)
        if evidence is None:
            raise ValueError("--use_rule_evidence_features requires non-empty evidence.")
        evidence = torch.tanh(torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0))
        return torch.cat([z_summary, evidence], dim=1)

    def classify(
        self,
        x_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z_mu: torch.Tensor,
    ) -> torch.Tensor:
        evidence = self.rule_evidence(x_norm, recon_norm, z_mu)
        logits = self.classifier(self.classifier_input(x_norm, recon_norm, z_mu, evidence))
        if self.use_rule_energy_logits and self.rule_energy_logit_weight > 0:
            if self.pinn_loss is None or not hasattr(self.pinn_loss, "rule_logits_from_evidence"):
                raise ValueError("--use_rule_energy_logits requires rule_logits_from_evidence().")
            if evidence is None:
                raise ValueError("--use_rule_energy_logits requires non-empty evidence.")
            logits = logits + self.rule_energy_logit_weight * self.pinn_loss.rule_logits_from_evidence(evidence)
        return logits

    def forward_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        beta: float,
        cls_weight: float,
        class_weights: torch.Tensor | None = None,
        pinn_weight: float = 0.0,
        chain_weight: float = 0.0,
        chain_flow_weight: float = 0.005,
        chain_temp_weight: float = 1.0,
        chain_teacher_forcing: float = 0.15,
        physical_weight: float = 0.0,
        physical_align_weight: float = 0.1,
        physical_trend_weight: float = 1.0,
        physical_normal_weight: float = 0.1,
        physical_smooth_weight: float = 0.01,
        physical_profile_weight: float = 0.0,
        physical_profile_start: float = 0.95,
        physical_profile_end: float = 0.05,
        physical_normal_flat_weight: float = 0.0,
        physical_decorrelation_weight: float = 0.0,
        physical_restriction_l2_weight: float = 0.0,
        transition_aux_weight: float = 0.0,
        transition_delta_l2_weight: float = 0.0,
        transition_physical_consistency_weight: float = 0.0,
        boundary_loss_weight: float = 0.0,
        boundary_classes: tuple[int, ...] = (0, 7, 8),
        boundary_margin: float = 0.5,
        boundary_gate_temperature: float = 0.25,
        boundary_gate_mode: str = "soft_focus",
        boundary_pair_gate_margin: float = 1.0,
        boundary_pair_gate_prob: float = 0.15,
        boundary_evidence_threshold: float = 0.35,
        sample_weights: torch.Tensor | None = None,
    ):
        xs = x
        ys = x
        zs_inf_dists, y_1t, _x_1t, _y_1T, x_1T = self.dvae.inference_func(xs, ys)
        zs = zs_inf_dists.sample()
        z0 = self.dvae.initial_latent(ys, xs)
        zs_pri_dists = self.dvae.get_priors(z0, zs, y_1t, x_1T)
        y_dists = self.dvae.measure.get_dist(zs, y_1t, x_1T)

        nll = -y_dists.log_prob(ys).sum(1).mean()
        kl = dists.kl.kl_divergence(zs_inf_dists, zs_pri_dists).sum(1).mean()
        evidence_dists = self.dvae.measure.get_dist(zs_inf_dists.loc, y_1t, x_1T)
        rule_evidence = None
        if self.use_rule_evidence_features or self.use_rule_energy_logits or boundary_gate_mode == "evidence_conditioned":
            rule_evidence = self.rule_evidence(x, evidence_dists.loc, zs_inf_dists.loc)
        logits = self.classifier(self.classifier_input(x, evidence_dists.loc, zs_inf_dists.loc, rule_evidence))
        if self.use_rule_energy_logits:
            if self.pinn_loss is None or not hasattr(self.pinn_loss, "rule_logits_from_evidence"):
                raise ValueError("--use_rule_energy_logits requires rule_logits_from_evidence().")
            if rule_evidence is None:
                raise ValueError("--use_rule_energy_logits requires non-empty evidence.")
            logits = logits + self.rule_energy_logit_weight * self.pinn_loss.rule_logits_from_evidence(rule_evidence)
        if sample_weights is None:
            ce = nn.functional.cross_entropy(logits, labels, weight=class_weights)
        else:
            ce_per_sample = nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
            ce = (ce_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        boundary = torch.zeros((), device=x.device)
        if boundary_loss_weight > 0:
            boundary = self.boundary_logit_margin_loss(
                logits=logits,
                labels=labels,
                boundary_classes=boundary_classes,
                margin=boundary_margin,
                gate_temperature=boundary_gate_temperature,
                gate_mode=boundary_gate_mode,
                pair_gate_margin=boundary_pair_gate_margin,
                pair_gate_prob=boundary_pair_gate_prob,
                evidence=rule_evidence,
                evidence_names=tuple(getattr(self.pinn_loss, "evidence_names", ())) if self.pinn_loss is not None else (),
                evidence_threshold=boundary_evidence_threshold,
            )
        pinn = torch.zeros((), device=x.device)
        if self.pinn_loss is not None and pinn_weight > 0:
            pinn, _pinn_parts = self.pinn_loss(
                x_norm=x,
                recon_norm=y_dists.loc,
                z_mu=zs_inf_dists.loc,
                labels=labels,
            )
        chain = torch.zeros((), device=x.device)
        if self.chain_prior_aux is not None and chain_weight > 0:
            physical_future = None
            if self.chain_prior_aux.use_physical_a:
                if self.physical_aux is None:
                    raise ValueError("--chain_use_physical_a requires --use_latent_physical_aux.")
                a_hat = self.physical_aux.opening(zs_inf_dists.loc)
                target_start = self.chain_prior_aux.history_len + self.chain_prior_aux.lag
                physical_future = a_hat[:, target_start : target_start + self.chain_prior_aux.pred_len]
            chain, _chain_parts = self.chain_prior_aux(
                x_norm=x,
                z_mu=zs_inf_dists.loc,
                flow_weight=chain_flow_weight,
                temp_weight=chain_temp_weight,
                teacher_forcing=chain_teacher_forcing,
                physical_future=physical_future,
            )
        physical = torch.zeros((), device=x.device)
        if self.physical_aux is not None and physical_weight > 0:
            physical, _physical_parts = self.physical_aux.forward_with_labels(
                x_norm=x,
                z_mu=zs_inf_dists.loc,
                labels=labels,
                align_weight=physical_align_weight,
                trend_weight=physical_trend_weight,
                normal_weight=physical_normal_weight,
                smooth_weight=physical_smooth_weight,
                profile_weight=physical_profile_weight,
                profile_start=physical_profile_start,
                profile_end=physical_profile_end,
                normal_flat_weight=physical_normal_flat_weight,
                decorrelation_weight=physical_decorrelation_weight,
                restriction_l2_weight=physical_restriction_l2_weight,
            )
        transition_aux = torch.zeros((), device=x.device)
        if transition_aux_weight > 0 and hasattr(self.dvae.transition, "aux_loss"):
            physical_opening = self.physical_aux.opening(zs_inf_dists.loc) if self.physical_aux is not None else None
            transition_aux, _transition_parts = self.dvae.transition.aux_loss(
                zs_inf_dists.loc,
                physical_opening,
                delta_l2_weight=transition_delta_l2_weight,
                physical_consistency_weight=transition_physical_consistency_weight,
            )
        loss = (
            nll
            + beta * kl
            + cls_weight * ce
            + pinn_weight * pinn
            + chain_weight * chain
            + physical_weight * physical
            + transition_aux_weight * transition_aux
            + boundary_loss_weight * boundary
        )
        return (
            loss,
            nll.detach(),
            kl.detach(),
            ce.detach(),
            pinn.detach(),
            chain.detach(),
            physical.detach(),
            transition_aux.detach(),
            boundary.detach(),
            logits,
        )

    @staticmethod
    def boundary_logit_margin_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_classes: tuple[int, ...],
        margin: float,
        gate_temperature: float,
        gate_mode: str = "soft_focus",
        pair_gate_margin: float = 1.0,
        pair_gate_prob: float = 0.15,
        evidence: torch.Tensor | None = None,
        evidence_names: tuple[str, ...] = (),
        evidence_threshold: float = 0.35,
    ) -> torch.Tensor:
        focus = tuple(int(c) for c in boundary_classes)
        if len(focus) < 2:
            return logits.sum() * 0.0
        gate_temperature = max(float(gate_temperature), 1e-6)
        if gate_mode in {"conditional_pair", "evidence_conditioned"}:
            probs = F.softmax(logits.detach(), dim=1)
            evidence_gate_inputs: dict[str, torch.Tensor] = {}
            if gate_mode == "evidence_conditioned":
                if evidence is None:
                    raise ValueError("boundary_gate_mode=evidence_conditioned requires rule evidence.")
                evidence = torch.tanh(torch.nan_to_num(evidence.detach(), nan=0.0, posinf=0.0, neginf=0.0))
                name_to_idx = {name: idx for idx, name in enumerate(evidence_names)}

                def ev_col(name: str, default: float = 0.0) -> torch.Tensor:
                    idx = name_to_idx.get(name)
                    if idx is None:
                        return torch.full((evidence.shape[0],), float(default), dtype=evidence.dtype, device=evidence.device)
                    return evidence[:, idx]

                pck_open = ev_col("pck_open_mean", 0.5)
                dhsv_open = ev_col("dhsv_open_mean", 0.5)
                pck_drop = ev_col("pck_drop", 0.0)
                pck_low = ev_col("pck_low_evidence", 0.0)
                dhsv_low = ev_col("dhsv_low_evidence", 0.0)
                pck_fault_score = (
                    F.relu(pck_drop)
                    + F.relu(pck_low)
                    + F.relu(0.5 - pck_open)
                ) / 3.0
                normal_stability_score = (
                    F.relu(dhsv_open - 0.5)
                    + F.relu(pck_open - 0.5)
                    + F.relu(0.5 - dhsv_low)
                    + F.relu(0.5 - pck_low)
                ) / 4.0
                threshold = float(evidence_threshold)
                evidence_gate_inputs = {
                    "pck_fault": torch.sigmoid((pck_fault_score - threshold) / gate_temperature),
                    "normal_stable": torch.sigmoid((normal_stability_score - threshold) / gate_temperature),
                    "not_normal_stable": torch.sigmoid((threshold - normal_stability_score) / gate_temperature),
                }
            pair_terms = []
            pair_specs = [(0, 7), (0, 8), (7, 0), (8, 0)]
            for target_cls, rival_cls in pair_specs:
                if target_cls not in focus or rival_cls not in focus:
                    continue
                mask = labels == target_cls
                if not bool(mask.any()):
                    continue
                gap = logits[mask, target_cls] - logits[mask, rival_cls]
                rival_prob = probs[mask, rival_cls]
                margin_gate = torch.sigmoid((float(pair_gate_margin) - gap.detach()) / gate_temperature)
                prob_gate = torch.sigmoid((rival_prob - float(pair_gate_prob)) / gate_temperature)
                logit_gate = torch.maximum(margin_gate, prob_gate)
                if gate_mode == "evidence_conditioned":
                    if {target_cls, rival_cls} == {0, 7}:
                        evidence_gate = evidence_gate_inputs["pck_fault"][mask]
                    elif target_cls == 0 and rival_cls == 8:
                        evidence_gate = evidence_gate_inputs["normal_stable"][mask]
                    elif target_cls == 8 and rival_cls == 0:
                        evidence_gate = evidence_gate_inputs["not_normal_stable"][mask]
                    else:
                        evidence_gate = torch.ones_like(logit_gate)
                    gate = evidence_gate * logit_gate
                else:
                    gate = logit_gate
                pair_terms.append(gate * F.softplus(float(margin) - gap))
            if not pair_terms:
                return logits.sum() * 0.0
            return torch.cat(pair_terms).mean()
        if gate_mode != "soft_focus":
            raise ValueError(f"Unknown boundary gate mode: {gate_mode}")
        focus_tensor = torch.as_tensor(focus, dtype=torch.long, device=logits.device)
        in_focus = (labels.unsqueeze(1) == focus_tensor.unsqueeze(0)).any(dim=1)
        if not bool(in_focus.any()):
            return logits.sum() * 0.0

        selected_logits = logits[in_focus]
        selected_labels = labels[in_focus]
        target_logits = selected_logits.gather(1, selected_labels.unsqueeze(1)).squeeze(1)
        focus_logits = selected_logits.index_select(1, focus_tensor)
        same_focus = selected_labels.unsqueeze(1) == focus_tensor.unsqueeze(0)
        rival_logits = focus_logits.masked_fill(same_focus, float("-inf"))
        rival_lse = torch.logsumexp(rival_logits, dim=1)
        gap = target_logits - rival_lse

        gate = torch.sigmoid((float(margin) - gap.detach()) / gate_temperature)
        return (gate * F.softplus(float(margin) - gap)).mean()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        xs = x
        ys = x
        zs_inf_dists, y_1t, _x_1t, _y_1T, x_1T = self.dvae.inference_func(xs, ys)
        recon_dists = self.dvae.measure.get_dist(zs_inf_dists.loc, y_1t, x_1T)
        return self.classify(x, recon_dists.loc, zs_inf_dists.loc)

    @torch.no_grad()
    def chain_prior_metrics(
        self,
        x: torch.Tensor,
        chain_flow_weight: float,
        chain_temp_weight: float,
    ) -> dict[str, float]:
        if self.chain_prior_aux is None:
            return {}
        xs = x
        ys = x
        zs_inf_dists, *_ = self.dvae.inference_func(xs, ys)
        physical_future = None
        if self.chain_prior_aux.use_physical_a:
            if self.physical_aux is None:
                raise ValueError("chain prior uses physical a, but physical_aux is not attached.")
            a_hat = self.physical_aux.opening(zs_inf_dists.loc)
            target_start = self.chain_prior_aux.history_len + self.chain_prior_aux.lag
            physical_future = a_hat[:, target_start : target_start + self.chain_prior_aux.pred_len]
        was_training = self.chain_prior_aux.training
        self.chain_prior_aux.eval()
        _loss, parts = self.chain_prior_aux(
            x_norm=x,
            z_mu=zs_inf_dists.loc,
            flow_weight=chain_flow_weight,
            temp_weight=chain_temp_weight,
            teacher_forcing=0.0,
            physical_future=physical_future,
        )
        if was_training:
            self.chain_prior_aux.train()
        return {key: float(value.cpu()) for key, value in parts.items()}

    @torch.no_grad()
    def structured_transition_metrics(self, x: torch.Tensor) -> dict[str, float]:
        if not hasattr(self.dvae.transition, "aux_loss"):
            return {}
        xs = x
        ys = x
        zs_inf_dists, y_1t, _x_1t, _y_1T, x_1T = self.dvae.inference_func(xs, ys)
        zs = zs_inf_dists.loc
        z0 = self.dvae.initial_latent(ys, xs)
        _zs_pri_dists = self.dvae.get_priors(z0, zs, y_1t, x_1T)
        physical_opening = self.physical_aux.opening(zs_inf_dists.loc) if self.physical_aux is not None else None
        _loss, parts = self.dvae.transition.aux_loss(
            zs_inf_dists.loc,
            physical_opening,
            delta_l2_weight=1.0,
            physical_consistency_weight=1.0,
        )
        return {key: float(value.cpu()) for key, value in parts.items()}

    @torch.no_grad()
    def physical_metrics(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        physical_align_weight: float,
        physical_trend_weight: float,
        physical_normal_weight: float,
        physical_smooth_weight: float,
        physical_profile_weight: float,
        physical_profile_start: float,
        physical_profile_end: float,
        physical_normal_flat_weight: float,
        physical_decorrelation_weight: float,
        physical_restriction_l2_weight: float,
    ) -> dict[str, float]:
        if self.physical_aux is None:
            return {}
        xs = x
        ys = x
        zs_inf_dists, *_ = self.dvae.inference_func(xs, ys)
        _loss, parts = self.physical_aux.forward_with_labels(
            x_norm=x,
            z_mu=zs_inf_dists.loc,
            labels=labels,
            align_weight=physical_align_weight,
            trend_weight=physical_trend_weight,
            normal_weight=physical_normal_weight,
            smooth_weight=physical_smooth_weight,
            profile_weight=physical_profile_weight,
            profile_start=physical_profile_start,
            profile_end=physical_profile_end,
            normal_flat_weight=physical_normal_flat_weight,
            decorrelation_weight=physical_decorrelation_weight,
            restriction_l2_weight=physical_restriction_l2_weight,
        )
        return {key: float(value.cpu()) for key, value in parts.items()}


def build_model(
    xdim: int,
    zdim: int,
    hdim: int,
    paper_architecture: bool,
    pinn_loss: nn.Module | None = None,
    chain_prior_aux: ChainPriorAuxLoss | None = None,
    physical_aux: LatentPhysicalAuxLoss | None = None,
    classifier_extra_dim: int = 0,
    use_rule_evidence_features: bool = False,
    use_rule_energy_logits: bool = False,
    rule_energy_logit_weight: float = 0.0,
    use_structured_transition: bool = False,
    transition_latent_indices: list[int] | None = None,
    transition_hidden_dim: int = 16,
    transition_residual_scale: float = 0.05,
) -> DVAEFaultDiagnoser:
    ydim = xdim
    x_encoder = RNNEncoder(xdim, hdim)
    y_encoder = RNNEncoder(ydim, hdim)
    if use_structured_transition:
        if transition_latent_indices is None:
            transition_latent_indices = list(range(zdim))
        transition = StructuredRestrictionTransition(
            zdim,
            hdim,
            0,
            x_encoder.hdim,
            latent_indices=transition_latent_indices,
            hidden_dim=transition_hidden_dim,
            residual_scale=transition_residual_scale,
        )
    else:
        transition = MLPTransition(zdim, hdim, 0, x_encoder.hdim)
    measure = MLPMeasure(ydim, zdim, hdim, y_encoder.hdim, x_encoder.hdim)
    inference = RNNInference(zdim, hdim, y_encoder.hdim, x_encoder.hdim)
    initializer = ControlInitializer(hdim, zdim, 0, xdim)
    dvae = DVAE(inference, transition, measure, y_encoder, x_encoder, initializer)
    configure_paper_architecture(dvae, paper_architecture)
    return DVAEFaultDiagnoser(
        dvae,
        LatentClassifier(zdim + classifier_extra_dim, hdim, 10),
        pinn_loss=pinn_loss,
        chain_prior_aux=chain_prior_aux,
        physical_aux=physical_aux,
        use_rule_evidence_features=use_rule_evidence_features,
        use_rule_energy_logits=use_rule_energy_logits,
        rule_energy_logit_weight=rule_energy_logit_weight,
    )


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    preds = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    accuracy = float((preds == y).mean()) if len(y) else 0.0
    f1s = []
    recalls = []
    per_class = {}
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y, preds):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion[int(true_label), int(pred_label)] += 1
    for cls in range(num_classes):
        tp = np.sum((preds == cls) & (y == cls))
        fp = np.sum((preds == cls) & (y != cls))
        fn = np.sum((preds != cls) & (y == cls))
        support = np.sum(y == cls)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if support > 0:
            f1s.append(f1)
            recalls.append(recall)
        per_class[str(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(model: DVAEFaultDiagnoser, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    for x, labels in loader:
        x = x.to(device).float()
        labels = labels.to(device)
        logits_all.append(model.predict(x).cpu())
        labels_all.append(labels.cpu())
    return metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), num_classes)


def evaluate_chain_prior(
    model: DVAEFaultDiagnoser,
    loader: DataLoader,
    device: torch.device,
    chain_flow_weight: float,
    chain_temp_weight: float,
) -> dict[str, float]:
    if model.chain_prior_aux is None:
        return {}
    model.eval()
    totals: dict[str, list[float]] = {}
    for x, _labels in loader:
        x = x.to(device).float()
        batch_metrics = model.chain_prior_metrics(x, chain_flow_weight, chain_temp_weight)
        for key, value in batch_metrics.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def evaluate_physical_aux(
    model: DVAEFaultDiagnoser,
    loader: DataLoader,
    device: torch.device,
    physical_align_weight: float,
    physical_trend_weight: float,
    physical_normal_weight: float,
    physical_smooth_weight: float,
    physical_profile_weight: float,
    physical_profile_start: float,
    physical_profile_end: float,
    physical_normal_flat_weight: float,
    physical_decorrelation_weight: float,
    physical_restriction_l2_weight: float,
) -> dict[str, float]:
    if model.physical_aux is None:
        return {}
    model.eval()
    totals: dict[str, list[float]] = {}
    for x, labels in loader:
        x = x.to(device).float()
        labels = labels.to(device)
        batch_metrics = model.physical_metrics(
            x,
            labels,
            physical_align_weight,
            physical_trend_weight,
            physical_normal_weight,
            physical_smooth_weight,
            physical_profile_weight,
            physical_profile_start,
            physical_profile_end,
            physical_normal_flat_weight,
            physical_decorrelation_weight,
            physical_restriction_l2_weight,
        )
        for key, value in batch_metrics.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def evaluate_structured_transition(
    model: DVAEFaultDiagnoser,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    if not hasattr(model.dvae.transition, "aux_loss"):
        return {}
    model.eval()
    totals: dict[str, list[float]] = {}
    for x, _labels in loader:
        x = x.to(device).float()
        batch_metrics = model.structured_transition_metrics(x)
        for key, value in batch_metrics.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def class_weights_for_dataset(
    dataset: ThreeWWindowDataset,
    num_classes: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    labels = np.array([label for _path, _start, label in dataset.index], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def paths_to_relative(paths: list[Path], data_root: Path, sort_paths: bool = False) -> list[str]:
    ordered_paths = sorted(paths) if sort_paths else paths
    return [str(path.relative_to(data_root)).replace("\\", "/") for path in ordered_paths]


def paths_sha256(paths: list[Path], data_root: Path) -> str:
    rel_paths = paths_to_relative(paths, data_root, sort_paths=True)
    return hashlib.sha256("\n".join(rel_paths).encode("utf-8")).hexdigest()


def parse_cols(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip() or value.strip().lower() in {"all", "none"}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_hard_boundary_mask(path: Path | None, dataset_len: int, device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"hard boundary file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    indices = data.get("train", {}).get("hard_indices")
    if indices is None:
        indices = data.get("train_hard_indices", [])
    valid_indices = sorted({int(idx) for idx in indices if 0 <= int(idx) < dataset_len})
    mask = torch.zeros(dataset_len, dtype=torch.bool, device=device)
    if valid_indices:
        mask[torch.as_tensor(valid_indices, dtype=torch.long, device=device)] = True
    return mask


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        elif (
            key in current
            and key.endswith(".weight")
            and current[key].ndim == 2
            and value.ndim == 2
            and current[key].shape[0] == value.shape[0]
            and current[key].shape[1] > value.shape[1]
        ):
            expanded = current[key].clone()
            expanded[:, : value.shape[1]] = value
            compatible[key] = expanded
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


def install_pathlib_pickle_compat() -> None:
    if "pathlib._local" in sys.modules:
        return
    pathlib_local = types.ModuleType("pathlib._local")
    pathlib_local.Path = pathlib.Path
    pathlib_local.PosixPath = pathlib.PosixPath
    pathlib_local.WindowsPath = pathlib.WindowsPath
    pathlib_local.PurePath = pathlib.PurePath
    pathlib_local.PurePosixPath = pathlib.PurePosixPath
    pathlib_local.PureWindowsPath = pathlib.PureWindowsPath
    sys.modules["pathlib._local"] = pathlib_local


def load_split(split_file: Path, data_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    split = json.loads(split_file.read_text(encoding="utf-8"))
    return (
        [data_root / path for path in split["train"]],
        [data_root / path for path in split["val"]],
        [data_root / path for path in split["test"]],
    )


def save_split(
    split_file: Path,
    data_root: Path,
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    seed: int,
    max_files_per_class: int | None,
) -> None:
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split = {
        "description": "Fixed instance-level split for 3W fault diagnosis experiments.",
        "data_root": str(data_root),
        "seed": seed,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "max_files_per_class": max_files_per_class,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "test_sha256": paths_sha256(test_paths, data_root),
        "order_note": "The train/val/test lists preserve the original order used by window sampling.",
        "train": paths_to_relative(train_paths, data_root),
        "val": paths_to_relative(val_paths, data_root),
        "test": paths_to_relative(test_paths, data_root),
    }
    split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("../3w_dataset_2.0.0"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split_file", type=Path, default=Path("splits/3w_seed42_max80_val15_test15.json"))
    parser.add_argument("--ignore_split_file", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max_files_per_class", type=int, default=30)
    parser.add_argument("--max_windows_per_file", type=int, default=8)
    parser.add_argument("--cache_size", type=int, default=32)
    parser.add_argument("--shuffle_windows", action="store_true")
    parser.add_argument("--clip_norm", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L2", type=float, default=1e-5)
    parser.add_argument("--chain_weight_decay", type=float)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--class_weighting", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--zdim", type=int, default=8)
    parser.add_argument("--hdim", type=int, default=64)
    parser.add_argument("--paper_architecture", action="store_true")
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate_checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bsw_pinn", action="store_true")
    parser.add_argument("--use_dhsv_pinn", action="store_true")
    parser.add_argument("--use_unified_physical_rule", action="store_true")
    parser.add_argument("--use_rule_evidence_features", action="store_true")
    parser.add_argument("--use_rule_energy_logits", action="store_true")
    parser.add_argument("--rule_energy_logit_weight", type=float, default=0.0)
    parser.add_argument("--pinn_root", type=Path, default=Path("../../DVAE_PINN"))
    parser.add_argument("--pinn_weight", type=float, default=1e-3)
    parser.add_argument("--pinn_trend_weight", type=float, default=1.0)
    parser.add_argument("--pinn_pressure_order_weight", type=float, default=0.0)
    parser.add_argument("--pinn_density_pressure_weight", type=float, default=0.0)
    parser.add_argument("--pinn_normal_smooth_weight", type=float, default=0.0)
    parser.add_argument("--dhsv_closure_weight", type=float, default=1.0)
    parser.add_argument("--dhsv_pressure_split_weight", type=float, default=1.0)
    parser.add_argument("--dhsv_flow_decay_weight", type=float, default=0.3)
    parser.add_argument("--dhsv_transient_direction_weight", type=float, default=0.5)
    parser.add_argument("--dhsv_normal_open_weight", type=float, default=0.1)
    parser.add_argument("--unified_bsw_trend_weight", type=float, default=0.2)
    parser.add_argument("--unified_bsw_density_pressure_weight", type=float, default=0.0)
    parser.add_argument("--unified_dhsv_closure_weight", type=float, default=1.0)
    parser.add_argument("--unified_pck_quick_closure_weight", type=float, default=1.0)
    parser.add_argument("--unified_pck_scaling_closure_weight", type=float, default=0.5)
    parser.add_argument("--unified_normal_open_weight", type=float, default=0.1)
    parser.add_argument("--unified_normal_smooth_weight", type=float, default=0.05)
    parser.add_argument("--use_chain_prior_aux", action="store_true")
    parser.add_argument("--chain_prior_weight", type=float, default=0.0)
    parser.add_argument("--chain_flow_weight", type=float, default=0.005)
    parser.add_argument("--chain_temp_weight", type=float, default=1.0)
    parser.add_argument("--chain_teacher_forcing", type=float, default=0.15)
    parser.add_argument("--chain_history_len", type=int, default=64)
    parser.add_argument("--chain_lag", type=int, default=4)
    parser.add_argument("--chain_pred_len", type=int, default=16)
    parser.add_argument("--chain_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_residual_scale", type=float, default=0.1)
    parser.add_argument("--chain_dropout", type=float, default=0.05)
    parser.add_argument("--chain_use_physical_a", action="store_true")
    parser.add_argument(
        "--chain_pressure_cols",
        type=str,
        default="P-PDG,P-TPT,P-MON-SDV-P,P-MON-CKP,P-JUS-CKP,P-JUS-BS,P-ANULAR",
    )
    parser.add_argument("--chain_flow_cols", type=str, default="QGL,QBS")
    parser.add_argument("--chain_temp_cols", type=str, default="T-TPT,T-JUS-CKP,T-MON-CKP,T-PDG")
    parser.add_argument("--use_latent_physical_aux", action="store_true")
    parser.add_argument("--physical_weight", type=float, default=0.0)
    parser.add_argument("--physical_align_weight", type=float, default=0.1)
    parser.add_argument("--physical_trend_weight", type=float, default=1.0)
    parser.add_argument("--physical_normal_weight", type=float, default=0.1)
    parser.add_argument("--physical_smooth_weight", type=float, default=0.01)
    parser.add_argument("--physical_profile_weight", type=float, default=0.0)
    parser.add_argument("--physical_profile_start", type=float, default=0.95)
    parser.add_argument("--physical_profile_end", type=float, default=0.05)
    parser.add_argument("--physical_normal_flat_weight", type=float, default=0.0)
    parser.add_argument("--physical_decorrelation_weight", type=float, default=0.0)
    parser.add_argument("--physical_restriction_l2_weight", type=float, default=0.0)
    parser.add_argument("--physical_col", type=str, default="ESTADO-DHSV")
    parser.add_argument("--physical_fault_label", type=int, default=2)
    parser.add_argument("--physical_normal_label", type=int, default=0)
    parser.add_argument("--physical_hidden_dim", type=int, default=16)
    parser.add_argument("--physical_mode", choices=["decrease", "increase"], default="decrease")
    parser.add_argument("--physical_trend_margin", type=float, default=0.2)
    parser.add_argument("--physical_sequence_head", action="store_true")
    parser.add_argument("--physical_latent_indices", type=str, default="all")
    parser.add_argument("--use_structured_transition", action="store_true")
    parser.add_argument("--transition_latent_indices", type=str, default="physical")
    parser.add_argument("--transition_hidden_dim", type=int, default=16)
    parser.add_argument("--transition_residual_scale", type=float, default=0.05)
    parser.add_argument("--transition_aux_weight", type=float, default=0.0)
    parser.add_argument("--transition_delta_l2_weight", type=float, default=0.1)
    parser.add_argument("--transition_physical_consistency_weight", type=float, default=1.0)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.0)
    parser.add_argument("--boundary_classes", type=str, default="0,7,8")
    parser.add_argument("--boundary_margin", type=float, default=0.5)
    parser.add_argument("--boundary_gate_temperature", type=float, default=0.25)
    parser.add_argument(
        "--boundary_gate_mode",
        choices=["soft_focus", "conditional_pair", "evidence_conditioned"],
        default="soft_focus",
    )
    parser.add_argument("--boundary_pair_gate_margin", type=float, default=1.0)
    parser.add_argument("--boundary_pair_gate_prob", type=float, default=0.15)
    parser.add_argument("--boundary_evidence_threshold", type=float, default=0.35)
    parser.add_argument("--hard_boundary_file", type=Path, default=None)
    parser.add_argument("--hard_boundary_weight", type=float, default=0.0)
    parser.add_argument("--freeze_dvae", action="store_true")
    parser.add_argument("--freeze_classifier", action="store_true")
    parser.add_argument("--freeze_pinn", action="store_true")
    parser.add_argument("--freeze_chain_prior", action="store_true")
    parser.add_argument("--freeze_physical_aux", action="store_true")
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--resume_state", type=Path, default=None)
    parser.add_argument("--save_resume_state", action="store_true")
    args = parser.parse_args()

    enabled_pinn_losses = sum([args.use_bsw_pinn, args.use_dhsv_pinn, args.use_unified_physical_rule])
    if enabled_pinn_losses > 1:
        raise ValueError(
            "Enable only one PINN loss at a time: "
            "--use_bsw_pinn, --use_dhsv_pinn, or --use_unified_physical_rule."
        )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    split_file = args.split_file
    if split_file is not None and not split_file.is_absolute():
        split_file = split_file.resolve()

    if split_file is not None and split_file.exists() and not args.ignore_split_file:
        train_paths, val_paths, test_paths = load_split(split_file, data_root)
        split_mode = "loaded"
    else:
        paths = list_parquet_files(data_root)
        train_paths, val_paths, test_paths = stratified_split_by_instance(
            paths,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
            max_files_per_class=args.max_files_per_class,
        )
        split_mode = "created"
        if split_file is not None:
            save_split(split_file, data_root, train_paths, val_paths, test_paths, args.seed, args.max_files_per_class)

    evaluation_checkpoint = None
    if args.evaluate_checkpoint is not None:
        install_pathlib_pickle_compat()
        evaluation_checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        stats = evaluation_checkpoint["stats"]
        args.epochs = 0
    else:
        stats = compute_normalization_stats(train_paths, seed=args.seed)

    train_ds = ThreeWWindowDataset(
        train_paths, args.T, args.stride, stats, task=args.task,
        max_windows_per_file=args.max_windows_per_file, seed=args.seed, cache_size=args.cache_size,
        clip_norm=args.clip_norm if args.clip_norm > 0 else None,
        return_index=bool(args.hard_boundary_file is not None and args.hard_boundary_weight > 0),
    )
    val_ds = ThreeWWindowDataset(
        val_paths, args.T, args.stride, stats, task=args.task,
        max_windows_per_file=args.max_windows_per_file, seed=args.seed, cache_size=args.cache_size,
        clip_norm=args.clip_norm if args.clip_norm > 0 else None,
    )
    test_ds = ThreeWWindowDataset(
        test_paths, args.T, args.stride, stats, task=args.task,
        max_windows_per_file=args.max_windows_per_file, seed=args.seed, cache_size=args.cache_size,
        clip_norm=args.clip_norm if args.clip_norm > 0 else None,
    )

    num_classes = 2 if args.task == "binary" else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=args.shuffle_windows, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, **loader_kwargs)
    hard_boundary_mask = None
    if args.hard_boundary_file is not None and args.hard_boundary_weight > 0:
        hard_boundary_mask = load_hard_boundary_mask(args.hard_boundary_file, len(train_ds), device)
    pinn_loss = None
    chain_prior_aux = None
    physical_aux = None
    if args.use_bsw_pinn:
        if args.task != "multiclass":
            raise ValueError("BSW PINN loss expects multiclass labels where fault 1 is BSW.")
        pinn_root = args.pinn_root.resolve()
        if str(pinn_root) not in sys.path:
            sys.path.append(str(pinn_root))
        from bsw_pinn_loss import BSWPhysicsLoss, BSWPhysicsWeights

        pinn_loss = BSWPhysicsLoss(
            FEATURE_COLUMNS,
            stats,
            zdim=args.zdim,
            weights=BSWPhysicsWeights(
                trend=args.pinn_trend_weight,
                pressure_order=args.pinn_pressure_order_weight,
                density_pressure=args.pinn_density_pressure_weight,
                normal_smooth=args.pinn_normal_smooth_weight,
            ),
        )
    if args.use_dhsv_pinn:
        if args.task != "multiclass":
            raise ValueError("DHSV PINN loss expects multiclass labels where fault 2 is DHSV.")
        pinn_root = args.pinn_root.resolve()
        if str(pinn_root) not in sys.path:
            sys.path.append(str(pinn_root))
        from dhsv_pinn_loss import DHSVPhysicsLoss, DHSVPhysicsWeights

        pinn_loss = DHSVPhysicsLoss(
            FEATURE_COLUMNS,
            stats,
            zdim=args.zdim,
            weights=DHSVPhysicsWeights(
                closure_trend=args.dhsv_closure_weight,
                pressure_split=args.dhsv_pressure_split_weight,
                flow_decay=args.dhsv_flow_decay_weight,
                transient_direction=args.dhsv_transient_direction_weight,
                normal_open=args.dhsv_normal_open_weight,
            ),
        )
    if args.use_unified_physical_rule:
        if args.task != "multiclass":
            raise ValueError("Unified physical rule loss expects multiclass labels.")
        pinn_root = args.pinn_root.resolve()
        if str(pinn_root) not in sys.path:
            sys.path.append(str(pinn_root))
        from unified_physical_rule_loss import UnifiedPhysicalRuleLoss, UnifiedPhysicalRuleWeights

        pinn_loss = UnifiedPhysicalRuleLoss(
            FEATURE_COLUMNS,
            stats,
            zdim=args.zdim,
            weights=UnifiedPhysicalRuleWeights(
                bsw_trend=args.unified_bsw_trend_weight,
                bsw_density_pressure=args.unified_bsw_density_pressure_weight,
                dhsv_closure=args.unified_dhsv_closure_weight,
                pck_quick_closure=args.unified_pck_quick_closure_weight,
                pck_scaling_closure=args.unified_pck_scaling_closure_weight,
                normal_open=args.unified_normal_open_weight,
                normal_smooth=args.unified_normal_smooth_weight,
            ),
        )
    if (args.use_rule_evidence_features or args.use_rule_energy_logits) and (
        pinn_loss is None or not hasattr(pinn_loss, "evidence_features")
    ):
        raise ValueError("rule evidence/logit fusion requires --use_unified_physical_rule.")
    classifier_extra_dim = int(getattr(pinn_loss, "evidence_dim", 0)) if args.use_rule_evidence_features else 0
    if args.use_chain_prior_aux:
        chain_prior_aux = ChainPriorAuxLoss(
            FEATURE_COLUMNS,
            zdim=args.zdim,
            pressure_cols=parse_cols(args.chain_pressure_cols),
            flow_cols=parse_cols(args.chain_flow_cols),
            temp_cols=parse_cols(args.chain_temp_cols),
            history_len=args.chain_history_len,
            lag=args.chain_lag,
            pred_len=args.chain_pred_len,
            hidden_dim=args.chain_hidden_dim,
            residual_scale=args.chain_residual_scale,
            dropout=args.chain_dropout,
            use_physical_a=args.chain_use_physical_a,
        )
    if args.use_latent_physical_aux:
        physical_aux = LatentPhysicalAuxLoss(
            FEATURE_COLUMNS,
            stats,
            zdim=args.zdim,
            observed_col=args.physical_col,
            fault_label=args.physical_fault_label,
            normal_label=args.physical_normal_label,
            hidden_dim=args.physical_hidden_dim,
            mode=args.physical_mode,
            trend_margin=args.physical_trend_margin,
            sequence_head=args.physical_sequence_head,
            latent_indices=parse_indices(args.physical_latent_indices),
        )
    model = build_model(
        xdim=27,
        zdim=args.zdim,
        hdim=args.hdim,
        paper_architecture=args.paper_architecture,
        pinn_loss=pinn_loss,
        chain_prior_aux=chain_prior_aux,
        physical_aux=physical_aux,
        classifier_extra_dim=classifier_extra_dim,
        use_rule_evidence_features=args.use_rule_evidence_features,
        use_rule_energy_logits=args.use_rule_energy_logits,
        rule_energy_logit_weight=args.rule_energy_logit_weight,
        use_structured_transition=args.use_structured_transition,
        transition_latent_indices=parse_indices(args.physical_latent_indices)
        if args.transition_latent_indices.strip().lower() == "physical"
        else parse_indices(args.transition_latent_indices),
        transition_hidden_dim=args.transition_hidden_dim,
        transition_residual_scale=args.transition_residual_scale,
    ).to(device)
    if args.task == "binary":
        model.classifier = LatentClassifier(args.zdim + classifier_extra_dim, args.hdim, 2).to(device)

    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint.resolve()
        if sys.platform.startswith("win"):
            pathlib.PosixPath = pathlib.WindowsPath
        install_pathlib_pickle_compat()
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected, skipped = load_compatible_state_dict(model, checkpoint["model"])
        print(
            "loaded init checkpoint: "
            f"{init_checkpoint}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, "
            f"skipped_shape_keys={len(skipped)}"
        )

    if args.freeze_dvae:
        for param in model.dvae.parameters():
            param.requires_grad = False
    if args.freeze_classifier:
        for param in model.classifier.parameters():
            param.requires_grad = False
    if args.freeze_pinn and model.pinn_loss is not None:
        for param in model.pinn_loss.parameters():
            param.requires_grad = False
    if args.freeze_chain_prior and model.chain_prior_aux is not None:
        for param in model.chain_prior_aux.parameters():
            param.requires_grad = False
    if args.freeze_physical_aux and model.physical_aux is not None:
        for param in model.physical_aux.parameters():
            param.requires_grad = False
    if any([args.freeze_dvae, args.freeze_classifier, args.freeze_pinn, args.freeze_chain_prior, args.freeze_physical_aux]):
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        print(f"frozen modules set; trainable_params={trainable}, total_params={total_params}")

    if args.chain_weight_decay is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    else:
        if model.chain_prior_aux is None or args.chain_weight_decay < 0:
            raise ValueError("A nonnegative chain_weight_decay requires a chain predictor.")
        primary = [parameter for name, parameter in model.named_parameters()
                   if not name.startswith("chain_prior_aux.")]
        optimizer = torch.optim.Adam([
            {"params": primary, "weight_decay": args.L2, "group_name": "primary"},
            {"params": model.chain_prior_aux.parameters(), "weight_decay": args.chain_weight_decay,
             "group_name": "chain"},
        ], lr=args.lr, weight_decay=args.L2)

    def chain_norm():
        if model.chain_prior_aux is None:
            return 0.0
        return sum(p.detach().double().square().sum().item()
                   for p in model.chain_prior_aux.parameters()) ** 0.5

    if args.chain_weight_decay is not None and not evaluation_checkpoint:
        initial_chain_norm = chain_norm()
    class_weights = None
    if args.class_weighting in {"balanced", "sqrt_balanced"}:
        class_weights = class_weights_for_dataset(train_ds, num_classes, device, args.class_weighting)
        print("class weights:", [round(float(i), 4) for i in class_weights.detach().cpu()])
    best_val = None if evaluation_checkpoint is not None else -1.0
    epochs_since_improvement = 0
    run_name = args.run_name or f"dvae_3w_{args.task}"
    model_path = args.evaluate_checkpoint or (args.output_dir / f"{run_name}.pth")
    resume_state_path = args.resume_state.resolve() if args.resume_state is not None else None
    start_epoch = 1
    if resume_state_path is not None and resume_state_path.exists() and args.epochs > 0:
        if sys.platform.startswith("win"):
            pathlib.PosixPath = pathlib.WindowsPath
        install_pathlib_pickle_compat()
        resume_state = torch.load(resume_state_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_val = float(resume_state.get("best_val", best_val))
        epochs_since_improvement = int(resume_state.get("epochs_since_improvement", epochs_since_improvement))
        if "stats" in resume_state:
            stats = resume_state["stats"]
        print(
            "resumed training state: "
            f"{resume_state_path}, completed_epoch={start_epoch - 1}, best_val_macro_f1={best_val:.4f}"
        )
    started = time.time()

    print(f"device: {device}")
    print(f"split: {split_mode} {split_file}")
    print(f"test split sha256: {paths_sha256(test_paths, data_root)}")
    print(f"windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"task: {args.task}, classes: {num_classes}")
    if args.use_bsw_pinn:
        print(
            "bsw pinn enabled: "
            f"lambda={args.pinn_weight}, "
            f"trend={args.pinn_trend_weight}, "
            f"pressure_order={args.pinn_pressure_order_weight}, "
            f"density_pressure={args.pinn_density_pressure_weight}, "
            f"normal_smooth={args.pinn_normal_smooth_weight}"
        )
    if args.use_dhsv_pinn:
        print(
            "dhsv pinn enabled: "
            f"lambda={args.pinn_weight}, "
            f"closure={args.dhsv_closure_weight}, "
            f"pressure_split={args.dhsv_pressure_split_weight}, "
            f"flow_decay={args.dhsv_flow_decay_weight}, "
            f"transient_direction={args.dhsv_transient_direction_weight}, "
            f"normal_open={args.dhsv_normal_open_weight}"
        )
    if args.use_unified_physical_rule:
        print(
            "unified physical rule enabled: "
            f"lambda={args.pinn_weight}, "
            f"bsw_trend={args.unified_bsw_trend_weight}, "
            f"bsw_density_pressure={args.unified_bsw_density_pressure_weight}, "
            f"dhsv_closure={args.unified_dhsv_closure_weight}, "
            f"pck_quick_closure={args.unified_pck_quick_closure_weight}, "
            f"pck_scaling_closure={args.unified_pck_scaling_closure_weight}, "
            f"normal_open={args.unified_normal_open_weight}, "
            f"normal_smooth={args.unified_normal_smooth_weight}"
        )
    if args.use_rule_evidence_features:
        print(f"rule evidence features enabled: dim={classifier_extra_dim}")
    if args.use_rule_energy_logits:
        print(f"rule energy logit fusion enabled: weight={args.rule_energy_logit_weight}")
    if args.use_chain_prior_aux:
        print(
            "chain prior auxiliary enabled: "
            f"lambda={args.chain_prior_weight}, "
            f"flow_weight={args.chain_flow_weight}, "
            f"temp_weight={args.chain_temp_weight}, "
            f"teacher_forcing={args.chain_teacher_forcing}, "
            f"history_len={args.chain_history_len}, lag={args.chain_lag}, pred_len={args.chain_pred_len}, "
            f"use_physical_a={args.chain_use_physical_a}"
        )
        print(
            "chain prior cols: "
            f"P={parse_cols(args.chain_pressure_cols)}, "
            f"Q={parse_cols(args.chain_flow_cols)}, "
            f"T={parse_cols(args.chain_temp_cols)}"
        )
    if args.use_latent_physical_aux:
        print(
            "latent physical auxiliary enabled: "
            f"lambda={args.physical_weight}, col={args.physical_col}, "
            f"fault_label={args.physical_fault_label}, mode={args.physical_mode}, "
            f"trend_margin={args.physical_trend_margin}, "
            f"profile_weight={args.physical_profile_weight}, "
            f"normal_flat={args.physical_normal_flat_weight}, "
            f"sequence_head={args.physical_sequence_head}, "
            f"latent_indices={parse_indices(args.physical_latent_indices) or 'all'}, "
            f"decorrelation={args.physical_decorrelation_weight}, "
            f"restriction_l2={args.physical_restriction_l2_weight}"
        )
    if args.use_structured_transition:
        transition_indices = (
            parse_indices(args.physical_latent_indices)
            if args.transition_latent_indices.strip().lower() == "physical"
            else parse_indices(args.transition_latent_indices)
        )
        print(
            "structured restriction transition enabled: "
            f"latent_indices={transition_indices or 'all'}, "
            f"hidden_dim={args.transition_hidden_dim}, "
            f"residual_scale={args.transition_residual_scale}, "
            f"aux_lambda={args.transition_aux_weight}, "
            f"delta_l2={args.transition_delta_l2_weight}, "
            f"physical_consistency={args.transition_physical_consistency_weight}"
        )
    if args.task == "multiclass":
        print("class names:", json.dumps(CLASS_NAMES, ensure_ascii=False))
    if hard_boundary_mask is not None:
        print(
            "hard boundary CE weighting enabled: "
            f"file={args.hard_boundary_file}, "
            f"alpha={args.hard_boundary_weight}, "
            f"hard_train_windows={int(hard_boundary_mask.sum().item())}/{len(train_ds)}"
        )
    boundary_classes = tuple(parse_indices(args.boundary_classes) or [0, 7, 8])
    if len(boundary_classes) < 2:
        raise ValueError("--boundary_classes must contain at least two class ids.")
    if args.boundary_loss_weight > 0:
        print(
            "boundary correction loss enabled: "
            f"lambda={args.boundary_loss_weight}, "
            f"classes={boundary_classes}, "
            f"margin={args.boundary_margin}, "
            f"gate_temperature={args.boundary_gate_temperature}, "
            f"gate_mode={args.boundary_gate_mode}, "
            f"pair_gate_margin={args.boundary_pair_gate_margin}, "
            f"pair_gate_prob={args.boundary_pair_gate_prob}, "
            f"evidence_threshold={args.boundary_evidence_threshold}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        nlls = []
        kls = []
        ces = []
        pinns = []
        chains = []
        physicals = []
        transitions = []
        boundaries = []
        for batch in train_loader:
            if len(batch) == 3:
                x, labels, sample_indices = batch
            else:
                x, labels = batch
                sample_indices = None
            x = x.to(device).float()
            labels = labels.to(device)
            sample_weights = None
            if hard_boundary_mask is not None and sample_indices is not None and args.hard_boundary_weight > 0:
                hard_batch = hard_boundary_mask[sample_indices.to(device)]
                sample_weights = torch.ones(labels.shape[0], dtype=torch.float32, device=device)
                sample_weights = sample_weights + hard_batch.to(dtype=torch.float32) * float(args.hard_boundary_weight)
            optimizer.zero_grad()
            loss, nll, kl, ce, pinn, chain, physical, transition_aux, boundary, _logits = model.forward_loss(
                x,
                labels,
                args.beta,
                args.cls_weight,
                class_weights,
                args.pinn_weight,
                args.chain_prior_weight,
                args.chain_flow_weight,
                args.chain_temp_weight,
                args.chain_teacher_forcing,
                args.physical_weight,
                args.physical_align_weight,
                args.physical_trend_weight,
                args.physical_normal_weight,
                args.physical_smooth_weight,
                args.physical_profile_weight,
                args.physical_profile_start,
                args.physical_profile_end,
                args.physical_normal_flat_weight,
                args.physical_decorrelation_weight,
                args.physical_restriction_l2_weight,
                args.transition_aux_weight,
                args.transition_delta_l2_weight,
                args.transition_physical_consistency_weight,
                args.boundary_loss_weight,
                boundary_classes,
                args.boundary_margin,
                args.boundary_gate_temperature,
                args.boundary_gate_mode,
                args.boundary_pair_gate_margin,
                args.boundary_pair_gate_prob,
                args.boundary_evidence_threshold,
                sample_weights,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            nlls.append(float(nll.cpu()))
            kls.append(float(kl.cpu()))
            ces.append(float(ce.cpu()))
            pinns.append(float(pinn.cpu()))
            chains.append(float(chain.cpu()))
            physicals.append(float(physical.cpu()))
            transitions.append(float(transition_aux.cpu()))
            boundaries.append(float(boundary.cpu()))

        val_metrics = evaluate(model, val_loader, device, num_classes)
        improved = val_metrics["macro_f1"] > best_val
        if improved:
            best_val = val_metrics["macro_f1"]
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if improved or args.save_last:
            torch.save({"model": model.state_dict(), "args": vars(args), "stats": stats}, model_path)
            saved = " saved"
        else:
            saved = ""
        if args.save_resume_state and resume_state_path is not None:
            resume_state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "epochs_since_improvement": epochs_since_improvement,
                    "args": vars(args),
                    "stats": stats,
                },
                resume_state_path,
            )
        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} nll={np.mean(nlls):.4f} "
            f"kl={np.mean(kls):.4f} ce={np.mean(ces):.4f} "
            f"pinn={np.mean(pinns):.4f} chain={np.mean(chains):.4f} "
            f"phys={np.mean(physicals):.4f} trans={np.mean(transitions):.4f} "
            f"boundary={np.mean(boundaries):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}{saved}"
        )
        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(f"early stopping after {epoch} epochs; best val macro-F1={best_val:.4f}")
            break

    install_pathlib_pickle_compat()
    checkpoint = evaluation_checkpoint or torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, num_classes)
    test_chain_metrics = evaluate_chain_prior(
        model,
        test_loader,
        device,
        args.chain_flow_weight,
        args.chain_temp_weight,
    )
    test_physical_metrics = evaluate_physical_aux(
        model,
        test_loader,
        device,
        args.physical_align_weight,
        args.physical_trend_weight,
        args.physical_normal_weight,
        args.physical_smooth_weight,
        args.physical_profile_weight,
        args.physical_profile_start,
        args.physical_profile_end,
        args.physical_normal_flat_weight,
        args.physical_decorrelation_weight,
        args.physical_restriction_l2_weight,
    )
    test_transition_metrics = evaluate_structured_transition(model, test_loader, device)
    result = {
        "task": args.task,
        "runtime_minutes": (time.time() - started) / 60.0,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "best_val_macro_f1": best_val,
        "test_metrics": test_metrics,
        "test_chain_prior_metrics": test_chain_metrics,
        "test_physical_aux_metrics": test_physical_metrics,
        "test_structured_transition_metrics": test_transition_metrics,
        "model_path": str(model_path),
        "split_file": str(split_file) if split_file is not None else None,
        "split_mode": split_mode,
        "test_split_sha256": paths_sha256(test_paths, data_root),
    }
    if args.chain_weight_decay is not None:
        result["best_chain_parameter_norm"] = chain_norm()
        if not evaluation_checkpoint:
            result["initial_chain_parameter_norm"] = initial_chain_norm
        result["optimizer_groups"] = [
            {"name": group["group_name"], "lr": group["lr"], "weight_decay": group["weight_decay"]}
            for group in optimizer.param_groups
        ]
    result["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result_path = args.output_dir / f"{run_name}_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
