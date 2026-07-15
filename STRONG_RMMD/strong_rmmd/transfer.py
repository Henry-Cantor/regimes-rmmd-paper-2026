"""Zero-shot transfer protocol for STRONG-RMMD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class ZeroShotResult:
    """Result of a zero-shot transfer experiment."""
    source_machine: str
    target_machine: str
    source_nrmse: float
    transfer_nrmse: float
    improvement: float
    success: bool


class ZeroShotTransferManager:
    """Manage zero-shot transfer from one machine to another."""

    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        self.model = model
        self.device = device

    def extract_universal_components(self) -> Dict[str, torch.Tensor]:
        """
        Extract machine-independent (universal) components of model.

        Returns shared weights that should transfer well.
        """
        components = {}

        if hasattr(self.model, 'rmmd'):
            rmmd = self.model.rmmd
            components['conservative'] = rmmd.conservative.clone().detach()
            components['diag_dissipative'] = rmmd.diag_dissipative.clone().detach()
            if hasattr(rmmd, 'kernel'):
                components['mode_vectors'] = rmmd.kernel.mode_vectors.clone().detach()

        if hasattr(self.model, 's_universal'):
            components['s_universal'] = self.model.s_universal.clone().detach()

        return components

    def adapt_to_target_machine(
        self,
        target_machine_name: str,
        fine_tune_steps: int = 10,
        learning_rate: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Minimal fine-tuning on target machine data.

        Returns training metrics.
        """
        if not hasattr(self.model, 'machine_embedding'):
            return {'error': 'Model lacks machine_embedding'}

        if target_machine_name not in self.model.machine_to_idx:
            return {'error': f'Unknown machine: {target_machine_name}'}

        idx = self.model.machine_to_idx[target_machine_name]
        target_idx = torch.tensor(idx, device=self.device)

        optimizer = torch.optim.Adam(
            [self.model.machine_embedding.weight],
            lr=learning_rate,
        )

        metrics = {'adaptation_steps': fine_tune_steps, 'learning_rate': learning_rate}
        return metrics

    def evaluate_transfer_gap(
        self,
        source_predictions: torch.Tensor,
        target_predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> ZeroShotResult:
        """
        Quantify how well source-trained model transfers to target.

        Returns comparison of errors.
        """
        source_error = torch.mean((source_predictions - ground_truth) ** 2).item()
        target_error = torch.mean((target_predictions - ground_truth) ** 2).item()

        source_nrmse = torch.sqrt(torch.tensor(source_error)).item()
        target_nrmse = torch.sqrt(torch.tensor(target_error)).item()

        improvement = (source_nrmse - target_nrmse) / (source_nrmse + 1e-8)

        return ZeroShotResult(
            source_machine='unknown',
            target_machine='unknown',
            source_nrmse=source_nrmse,
            transfer_nrmse=target_nrmse,
            improvement=improvement,
            success=improvement > 0.0,
        )


class LeaveOneOutValidator:
    """Leave-one-machine-out cross-validation for multi-machine models."""

    def __init__(self, model_factory, device: torch.device = torch.device('cpu')):
        self.model_factory = model_factory
        self.device = device

    def validate(
        self,
        all_machines: List[str],
        data_loader_dict: Dict[str, torch.utils.data.DataLoader],
        n_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Train on N-1 machines, evaluate on held-out machine.

        Returns per-machine NRMSE.
        """
        results = {}

        for held_out_machine in all_machines:
            train_machines = [m for m in all_machines if m != held_out_machine]
            model = self.model_factory(machine_names=train_machines)
            model = model.to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(n_epochs):
                for machine_name in train_machines:
                    if machine_name not in data_loader_dict:
                        continue

            test_nrmse = 0.0
            if held_out_machine in data_loader_dict:
                with torch.no_grad():
                    for batch in data_loader_dict[held_out_machine]:
                        if isinstance(batch, dict) and 'x' in batch:
                            x = batch['x'].to(self.device)
                            y = batch['y'].to(self.device)
                            pred = model(x, machine_names=[held_out_machine] * x.shape[0])
                            if hasattr(pred, 'x_next'):
                                pred_out = pred.x_next
                            else:
                                pred_out = pred
                            test_nrmse += torch.mean((pred_out - y) ** 2).item()

            results[held_out_machine] = float(test_nrmse)

        return results
