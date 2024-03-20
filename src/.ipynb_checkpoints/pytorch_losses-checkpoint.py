import torch
import torch.nn as nn

def interpolate_values(input_array, pixel_coords):
    device = input_array.device
    height = input_array.shape[-2]
    width = input_array.shape[-1]
    samples = input_array.shape[0]
    points = pixel_coords.shape[-2]
    
    interpolated_values = torch.zeros(samples, points, device=device)

    # Get integer and fractional parts of pixel coords
    y_int = pixel_coords[..., 1].long()
    x_int = pixel_coords[..., 0].long()
    y_frac = pixel_coords[..., 1] - y_int.float()
    x_frac = pixel_coords[..., 0] - x_int.float()

    # Compute indices of neighbouring pixels
    y0 = torch.clamp(y_int, 0, height - 2)
    y1 = y0 + 1
    x0 = torch.clamp(x_int, 0, width - 2)
    x1 = x0 + 1

    for i in range(samples):
        values_y0x0 = input_array[i, 0, y0[i], x0[i]]
        values_y1x0 = input_array[i, 0, y1[i], x0[i]]
        values_y0x1 = input_array[i, 0, y0[i], x1[i]]
        values_y1x1 = input_array[i, 0, y1[i], x1[i]]

        # bilinear interpolation
        interpolated_values[i] = (
            values_y0x0 * (1 - y_frac[i]) * (1 - x_frac[i]) +
            values_y1x0 * y_frac[i] * (1 - x_frac[i]) +
            values_y0x1 * (1 - y_frac[i]) * x_frac[i] +
            values_y1x1 * y_frac[i] * x_frac[i]
        )

    return interpolated_values

def torch_tracked_mse_interp(y_pred, y_true):
    eps = 1e-7
    device = y_pred.device

    losses = torch.zeros(y_pred.shape[1], device=device)
    
    for i in range(y_pred.shape[1]):
        data = y_pred[:, i]
        
        target_coords = y_true[:, i, :, :-1]
        
        y_pred_loss = interpolate_values(data, target_coords)
        
        y_true_loss = y_true[:, i, :, -1]
        y_pred_loss *= (y_true_loss != 0).float()
        
        N_nz = torch.sum(y_true_loss != 0)
        N = N_nz + torch.sum(y_true_loss == 0)
        
        loss_loop = (N / (N_nz + eps)) * nn.MSELoss()(y_true_loss, y_pred_loss)
        
        losses[i] = loss_loop
    
    return torch.mean(losses)
