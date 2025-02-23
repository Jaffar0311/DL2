from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

def test_logging(logger: tb.SummaryWriter):
    """
    Logs dummy training loss and accuracy for training and validation.
    """
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # Training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = (epoch / 10.0) + torch.randn(10).mean().item()

            # Log training loss
            logger.add_scalar('train_loss', dummy_train_loss, global_step)
            
            # Save accuracy for averaging
            metrics["train_acc"].append(dummy_train_accuracy)

            global_step += 1
        
        # Log average training accuracy
        avg_train_accuracy = sum(metrics["train_acc"]) / len(metrics["train_acc"])
        logger.add_scalar('train_accuracy', avg_train_accuracy, global_step)

        # Validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = (epoch / 10.0) + torch.randn(10).mean().item()
            
            # Save validation accuracy for averaging
            metrics["val_acc"].append(dummy_validation_accuracy)

        # Log average validation accuracy
        avg_val_accuracy = sum(metrics["val_acc"]) / len(metrics["val_acc"])
        logger.add_scalar('val_accuracy', avg_val_accuracy, global_step)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)