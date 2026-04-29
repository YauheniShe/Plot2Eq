import torch
import torch.nn.functional as F


class HandDrawnAugmentation:
    def __init__(self, max_drift_scale=0.05, max_wobble_scale=0.01, p=0.8):
        """
        :param max_drift_scale: Максимальное плавное искажение
        :param max_wobble_scale: Максимальные мелкие неровности
        :param p: Вероятность применения аугментации (иногда полезно видеть идеальные данные)
        """
        self.max_drift_scale = max_drift_scale
        self.max_wobble_scale = max_wobble_scale
        self.p = p

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return points

        y = points[0:1, :].clone()
        mask = points[1:2, :].clone()
        seq_len = y.size(1)
        device = y.device

        min_strength = 0.5
        drift_strength = (
            torch.empty(1, device=device).uniform_(min_strength, 1.0).item()
        )
        wobble_strength = (
            torch.empty(1, device=device).uniform_(min_strength, 1.0).item()
        )

        current_drift = self.max_drift_scale * drift_strength
        current_wobble = self.max_wobble_scale * wobble_strength

        drift_freq = torch.randint(2, 6, (1,)).item()
        wobble_freq = torch.randint(10, 30, (1,)).item()

        low_freq_noise = torch.randn(1, 1, 1, drift_freq, device=device)  # type: ignore
        drift = F.interpolate(
            low_freq_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(1, seq_len)

        mid_freq_noise = torch.randn(1, 1, 1, wobble_freq, device=device)  # type: ignore
        wobble = F.interpolate(
            mid_freq_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(1, seq_len)

        total_distortion = (drift * current_drift) + (wobble * current_wobble)
        y = y + total_distortion * mask

        x_freq = torch.randint(5, 15, (1,)).item()
        x_noise = torch.randn(1, 1, 1, x_freq, device=device)  # type: ignore
        x_drift = F.interpolate(
            x_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(seq_len)

        envelope = F.interpolate(
            torch.rand(1, 1, 1, 3, device=device),
            size=(1, seq_len),
            mode="bicubic",
            align_corners=True,
        ).view(seq_len)
        x_drift = x_drift * envelope

        x_shift_amplitude = torch.rand(1, device=device).item() * 3.0 + 1.0
        new_indices = (
            torch.arange(seq_len, device=device).float() + x_drift * x_shift_amplitude
        )
        new_indices = torch.clamp(new_indices, 0, seq_len - 1)

        idx_floor = new_indices.long()
        idx_ceil = torch.clamp(idx_floor + 1, 0, seq_len - 1)
        weight = new_indices - idx_floor.float()

        mask_floor = mask[:, idx_floor]
        mask_ceil = mask[:, idx_ceil]

        valid_transition = mask_floor * mask_ceil

        idx_nearest = torch.round(new_indices).long()
        y_nearest = y[:, idx_nearest]

        y_interp = y[:, idx_floor] * (1.0 - weight) + y[:, idx_ceil] * weight

        y = torch.where(valid_transition == 1.0, y_interp, y_nearest)

        mask = mask[:, idx_nearest]

        if torch.rand(1).item() > 0.5:
            shear_factor = (torch.rand(1, device=device) - 0.5) * 0.1
            tiling = torch.linspace(-1, 1, seq_len, device=device) * shear_factor
            y = y + tiling * mask

        y_valid_mask = mask.bool().squeeze(0)

        if y_valid_mask.any():
            y_valid_points = y[0, y_valid_mask]
            min_val = torch.min(y_valid_points)
            max_val = torch.max(y_valid_points)

            if max_val > min_val:
                y[0, y_valid_mask] = (y[0, y_valid_mask] - min_val) / (
                    max_val - min_val
                )
            else:
                y[0, y_valid_mask] = 0.5

        return torch.cat((y, mask), dim=0)
