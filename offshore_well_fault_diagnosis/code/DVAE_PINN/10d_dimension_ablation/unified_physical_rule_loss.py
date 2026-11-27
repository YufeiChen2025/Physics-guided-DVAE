from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class UnifiedPhysicalRuleWeights:
    bsw_trend: float = 0.2
    bsw_density_pressure: float = 0.0
    dhsv_closure: float = 1.0
    pck_quick_closure: float = 1.0
    pck_scaling_closure: float = 0.5
    normal_open: float = 0.1
    normal_smooth: float = 0.05


class UnifiedPhysicalRuleLoss(nn.Module):
    """10D rule-evidence module with dimension ablation support.

    Ablation is controlled by environment variables so the existing training
    script can import this module through ``--pinn_root`` without being edited:

    - ``RULE_EVIDENCE_KEEP=water_rise,dhsv_drop``
    - ``RULE_EVIDENCE_DROP=dp_trend,pck_smooth``
    """

    full_evidence_names = [
        "water_rise",
        "water_smooth",
        "dhsv_drop",
        "dhsv_open_mean",
        "pck_drop",
        "pck_open_mean",
        "pck_smooth",
        "dhsv_low_evidence",
        "pck_low_evidence",
        "dp_trend",
    ]

    def __init__(
        self,
        feature_columns: list[str],
        stats: dict[str, torch.Tensor] | dict[str, object],
        zdim: int,
        weights: UnifiedPhysicalRuleWeights | None = None,
        normal_label: int = 0,
        bsw_label: int = 1,
        dhsv_label: int = 2,
        pck_quick_label: int = 6,
        pck_scaling_label: int = 7,
        opening_margin: float = 0.20,
        bsw_margin: float = 0.15,
        num_classes: int = 10,
        rule_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.col = {name: i for i, name in enumerate(self.feature_columns)}
        self.weights = weights or UnifiedPhysicalRuleWeights()
        self.normal_label = normal_label
        self.bsw_label = bsw_label
        self.dhsv_label = dhsv_label
        self.pck_quick_label = pck_quick_label
        self.pck_scaling_label = pck_scaling_label
        self.opening_margin = opening_margin
        self.bsw_margin = bsw_margin
        self.num_classes = num_classes
        self.rule_temperature = max(float(rule_temperature), 1e-6)

        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.water_cut_head = nn.Sequential(nn.Linear(zdim, zdim), nn.Tanh(), nn.Linear(zdim, 1))
        self.dhsv_opening_head = nn.Sequential(nn.Linear(zdim, zdim), nn.Tanh(), nn.Linear(zdim, 1))
        self.pck_opening_head = nn.Sequential(nn.Linear(zdim, zdim), nn.Tanh(), nn.Linear(zdim, 1))
        self.dhsv_state_mix_raw = nn.Parameter(torch.tensor(0.0))
        self.pck_state_mix_raw = nn.Parameter(torch.tensor(0.0))

        self.dp_wc_raw = nn.Parameter(torch.tensor(0.0))
        self.dp_bias = nn.Parameter(torch.tensor(0.0))
        self.dp_qgl = nn.Parameter(torch.tensor(0.0))
        self.dp_choke = nn.Parameter(torch.tensor(0.0))
        self.dp_gl_choke = nn.Parameter(torch.tensor(0.0))
        self.dp_temp = nn.Parameter(torch.tensor(0.0))

        self.keep_indices = self._parse_keep_indices()
        self.evidence_names = [self.full_evidence_names[i] for i in self.keep_indices]
        self.evidence_dim = len(self.evidence_names)
        if self.evidence_dim <= 0:
            raise ValueError("At least one rule-evidence dimension must be kept.")

        self.rule_head = nn.Linear(self.evidence_dim, self.num_classes)
        self._init_rule_head()

    def forward(
        self,
        x_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z_mu: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_norm.ndim != 3 or recon_norm.ndim != 3 or z_mu.ndim != 3:
            raise ValueError("x_norm, recon_norm, and z_mu must be [batch, time, dim].")

        evidence = self.evidence_features(x_norm=x_norm, recon_norm=recon_norm, z_mu=z_mu)
        rule_logits = self.rule_logits_from_evidence(evidence) / self.rule_temperature
        total = F.cross_entropy(rule_logits, labels)
        pred = rule_logits.argmax(dim=1)
        true_logit = rule_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        competing_logits = rule_logits.masked_fill(
            F.one_hot(labels, num_classes=self.num_classes).bool(),
            torch.finfo(rule_logits.dtype).min,
        )
        margin = true_logit - competing_logits.max(dim=1).values
        parts = {
            "pinn_total": total.detach(),
            "rule_energy_ce": total.detach(),
            "rule_energy_acc": (pred == labels).to(dtype=rule_logits.dtype).mean().detach(),
            "rule_energy_margin": margin.mean().detach(),
        }
        for idx, name in enumerate(self.evidence_names):
            parts[f"rule_{name}_mean"] = evidence[:, idx].detach().mean()
        return total, parts

    def rule_logits_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or evidence.shape[1] != self.evidence_dim:
            raise ValueError(f"evidence must be [batch, {self.evidence_dim}].")
        return self.rule_head(self._normalize_rule_evidence(evidence))

    def evidence_features(
        self,
        x_norm: torch.Tensor,
        recon_norm: torch.Tensor,
        z_mu: torch.Tensor,
    ) -> torch.Tensor:
        x_hat = self._unnormalize(recon_norm)
        water_cut = torch.sigmoid(self.water_cut_head(z_mu)).squeeze(-1)
        dhsv_open = self._effective_opening(
            x_hat,
            z_mu,
            observed_col="ESTADO-DHSV",
            head=self.dhsv_opening_head,
            mix_raw=self.dhsv_state_mix_raw,
            percent_scale=False,
        )
        pck_open = self._effective_opening(
            x_hat,
            z_mu,
            observed_col="ABER-CKP",
            head=self.pck_opening_head,
            mix_raw=self.pck_state_mix_raw,
            percent_scale=True,
        )

        water_first, water_last = self._quarter_means(water_cut)
        dhsv_first, dhsv_last = self._quarter_means(dhsv_open)
        pck_first, pck_last = self._quarter_means(pck_open)
        water_rise = water_last - water_first
        dhsv_drop = dhsv_first - dhsv_last
        pck_drop = pck_first - pck_last
        water_smooth = self._sequence_roughness(water_cut)
        pck_smooth = self._sequence_roughness(pck_open)

        dp_tubing = (
            self._feature_units(x_hat, "P-PDG") - self._feature_units(x_hat, "P-TPT")
        ) / self.std[self._idx("P-PDG")].clamp_min(1.0)
        dp_first, dp_last = self._quarter_means(dp_tubing)
        dp_trend = dp_last - dp_first

        full_features = torch.stack(
            [
                water_rise,
                water_smooth,
                dhsv_drop,
                dhsv_open.mean(dim=1),
                pck_drop,
                pck_open.mean(dim=1),
                pck_smooth,
                F.relu(0.85 - dhsv_open).mean(dim=1),
                F.relu(0.35 - pck_open).mean(dim=1),
                dp_trend,
            ],
            dim=1,
        )
        full_features = torch.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)
        index = torch.as_tensor(self.keep_indices, dtype=torch.long, device=full_features.device)
        return full_features.index_select(1, index)

    def _normalize_rule_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0))

    def _init_rule_head(self) -> None:
        full_weight = torch.zeros(self.num_classes, len(self.full_evidence_names))
        w = full_weight
        w[self.normal_label, 1] = -self.weights.normal_smooth
        w[self.normal_label, 3] = self.weights.normal_open
        w[self.normal_label, 5] = self.weights.normal_open
        w[self.normal_label, 7] = -self.weights.normal_open
        w[self.normal_label, 8] = -self.weights.normal_open
        w[self.normal_label, 9] = -0.05

        w[self.bsw_label, 0] = self.weights.bsw_trend
        w[self.bsw_label, 1] = 0.25 * self.weights.bsw_trend
        w[self.bsw_label, 9] = 0.10

        w[self.dhsv_label, 2] = self.weights.dhsv_closure
        w[self.dhsv_label, 3] = -0.25 * self.weights.dhsv_closure
        w[self.dhsv_label, 7] = self.weights.dhsv_closure

        w[5, 9] = 0.4
        w[5, 3] = 0.15
        w[5, 5] = 0.15
        w[5, 7] = -0.2
        w[5, 8] = -0.2

        w[self.pck_quick_label, 4] = self.weights.pck_quick_closure
        w[self.pck_quick_label, 6] = 0.25 * self.weights.pck_quick_closure
        w[self.pck_quick_label, 8] = self.weights.pck_quick_closure

        w[self.pck_scaling_label, 4] = 0.25 * self.weights.pck_scaling_closure
        w[self.pck_scaling_label, 6] = -0.25 * self.weights.pck_scaling_closure
        w[self.pck_scaling_label, 8] = self.weights.pck_scaling_closure

        w[8, 9] = 0.25
        w[8, 5] = 0.15
        w[9, 9] = 0.15

        keep = torch.as_tensor(self.keep_indices, dtype=torch.long)
        with torch.no_grad():
            self.rule_head.weight.copy_(full_weight.index_select(1, keep))
            self.rule_head.bias.zero_()

    def _parse_keep_indices(self) -> list[int]:
        keep_raw = os.environ.get("RULE_EVIDENCE_KEEP", "").strip()
        drop_raw = os.environ.get("RULE_EVIDENCE_DROP", "").strip()
        all_names = self.full_evidence_names
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}

        if keep_raw:
            keep_names = self._parse_name_list(keep_raw)
            unknown = [name for name in keep_names if name not in name_to_idx]
            if unknown:
                raise ValueError(f"Unknown RULE_EVIDENCE_KEEP names: {unknown}")
            keep = [name_to_idx[name] for name in keep_names]
        else:
            keep = list(range(len(all_names)))

        if drop_raw:
            drop_names = self._parse_name_list(drop_raw)
            unknown = [name for name in drop_names if name not in name_to_idx]
            if unknown:
                raise ValueError(f"Unknown RULE_EVIDENCE_DROP names: {unknown}")
            drop = {name_to_idx[name] for name in drop_names}
            keep = [idx for idx in keep if idx not in drop]

        return keep

    @staticmethod
    def _parse_name_list(value: str) -> list[str]:
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]

    def _unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std.view(1, 1, -1) + self.mean.view(1, 1, -1)

    def _idx(self, name: str) -> int:
        if name not in self.col:
            raise KeyError(f"Required 3W feature is missing: {name}")
        return self.col[name]

    def _feature_units(self, x_units: torch.Tensor, name: str) -> torch.Tensor:
        return torch.nan_to_num(x_units[..., self._idx(name)], nan=0.0, posinf=0.0, neginf=0.0)

    def _bounded_feature(self, x_units: torch.Tensor, name: str, percent_scale: bool) -> torch.Tensor:
        values = self._feature_units(x_units, name)
        if percent_scale:
            values = values / 100.0
        return torch.clamp(values, 0.0, 1.0)

    def _effective_opening(
        self,
        x_units: torch.Tensor,
        z_mu: torch.Tensor,
        observed_col: str,
        head: nn.Module,
        mix_raw: torch.Tensor,
        percent_scale: bool,
    ) -> torch.Tensor:
        latent_open = torch.sigmoid(head(z_mu)).squeeze(-1)
        observed_open = self._bounded_feature(x_units, observed_col, percent_scale=percent_scale)
        mix = torch.sigmoid(mix_raw)
        return mix * observed_open + (1.0 - mix) * latent_open

    def _quarter_means(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = max(values.shape[1] // 4, 1)
        return values[:, :q].mean(dim=1), values[:, -q:].mean(dim=1)

    def _sequence_roughness(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] < 2:
            return values.mean(dim=1) * 0.0
        diffs = values[:, 1:] - values[:, :-1]
        return diffs.square().mean(dim=1).sqrt()
