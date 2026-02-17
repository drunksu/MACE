#!/usr/bin/env python3
"""
Train conditional normalizing flow.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from mace_rl.utils.config import load_config
from mace_rl.utils.logging import setup_logging
from mace_rl.data.dataset import create_data_loaders
from mace_rl.models.flows import ConditionalRealNVP
from mace_rl.training.flow_trainer import FlowTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/flow.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    config = load_config(args.config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_file=config['data']['dataset_path'],
        val_file=None,
        batch_size=config['training']['batch_size'],
        val_split=config['training']['validation_split'],
        shuffle=True,
        state_prefix=config['data'].get('state_columns', 'state_'),
        action_column=config['data'].get('action_column', 'action'),
    )

    # Create model
    model = ConditionalRealNVP(
        input_dim=config['model']['input_dim'],
        context_dim=config['model']['context_dim'],
        num_transforms=config['model']['num_transforms'],
        hidden_dim=config['model']['hidden_dim'],
        num_blocks=config['model']['num_blocks'],
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm'],
    )

    # Optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )

    # Trainer
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        checkpoint_dir=config['checkpoint']['save_dir'],
        log_dir=config['logging']['log_dir'],
        experiment_name='flow',
        early_stopping_patience=config['training']['early_stopping_patience'],
    )

    # Train
    trainer.train(config['training']['epochs'])

    print("Training completed.")


if __name__ == '__main__':
    main()