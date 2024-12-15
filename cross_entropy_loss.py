# a utility function to calculate cross-entropy loss of any given batch

import torch
import torch.nn.functional as F

# this computes the loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

# this calculates the loss across all the batches in a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0 # starting point
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # iterates over all batches if no fixed num_batches specified
    else:
        # if num_batches > len(data_loader), reduce the num_batches to match len(data_loader) so that it still works
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # averaging the loss over all the batches
