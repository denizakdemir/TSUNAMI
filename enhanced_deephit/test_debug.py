import numpy as np
import torch
import torch.nn.functional as F

# Test sigmoid with extreme values
log_hazards = torch.tensor(np.random.uniform(-10, 10, (5, 10)), dtype=torch.float32)
print("Log hazards min/max:", log_hazards.min().item(), log_hazards.max().item())

hazards = torch.sigmoid(log_hazards)
print("Hazards min/max:", hazards.min().item(), hazards.max().item())

# Test if any value is outside [0, 1] bound
outside_bounds = ((hazards < 0) | (hazards > 1)).any()
print("Values outside [0, 1]:", outside_bounds.item())

# Test binary cross entropy
bce_targets = torch.zeros_like(hazards)
bce_mask = torch.ones_like(hazards)

try:
    bce_loss = F.binary_cross_entropy(
        hazards, 
        bce_targets, 
        reduction='none'
    )
    print("BCE loss succeeded")
except Exception as e:
    print("BCE loss failed:", str(e))