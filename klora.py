from typing import Optional, Union
import torch
from torch import nn

glo_count = 0


class KLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        weight_3_a: torch.Tensor = None,  # Lighting/Atmosphere LoRA (optional)
        weight_3_b: torch.Tensor = None,
        average_ratio: float = 1.0,       # Content vs Style ratio
        average_ratio_1_3: float = 1.0,   # Content vs Lighting ratio
        average_ratio_2_3: float = 1.0,   # Style vs Lighting ratio
        rank: int = 8,
        alpha: int = 1.5,
        beta: int = 0.5,
        gamma: float = 0.3,               # Lighting time scaling factor
        sum_timesteps: int = 28000,
        pattern: str = "s*",
        num_loras: int = 2,               # Number of LoRAs (2 or 3)
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        # 3rd LoRA (optional)
        self.weight_3_a = weight_3_a.to(device) if weight_3_a is not None else None
        self.weight_3_b = weight_3_b.to(device) if weight_3_b is not None else None
        self.average_ratio = average_ratio
        self.average_ratio_1_3 = average_ratio_1_3
        self.average_ratio_2_3 = average_ratio_2_3
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sum_timesteps = sum_timesteps
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"
        self.pattern = pattern
        self.num_loras = num_loras

    # select topk weights
    def get_klora_weight(self, timestep):
        sum_timesteps = self.sum_timesteps
        k = self.weight_1_a.shape[1] * self.weight_2_a.shape[1]
        alpha = self.alpha
        beta = self.beta
        avg_ratio = self.average_ratio

        # compute the sum of top k values
        time_ratio = timestep % sum_timesteps
        t = time_ratio / sum_timesteps  # normalized time (0 to 1)

        # Compute matrix products
        matrix1 = self.weight_1_a @ self.weight_1_b  # Content
        matrix2 = self.weight_2_a @ self.weight_2_b  # Style

        # Compute Top-K sums
        abs_matrix1 = torch.abs(matrix1)
        top_k_sum1 = torch.topk(abs_matrix1.flatten(), k)[0].sum()

        abs_matrix2 = torch.abs(matrix2)
        top_k_sum2 = torch.topk(abs_matrix2.flatten(), k)[0].sum()

        if self.num_loras == 2:
            # Original 2-LoRA binary selection (backward compatible)
            scale = alpha * time_ratio / sum_timesteps + beta
            if self.pattern == "s*":
                scale = scale % alpha

            top_k_sum1_scaled = top_k_sum1 / avg_ratio
            top_k_sum2_scaled = top_k_sum2 * scale

            temp_ratio = top_k_sum1_scaled / top_k_sum2_scaled
            if temp_ratio > 1:
                return matrix1
            else:
                return matrix2

        else:  # 3 LoRAs
            matrix3 = self.weight_3_a @ self.weight_3_b  # Lighting/Atmosphere
            abs_matrix3 = torch.abs(matrix3)
            top_k_sum3 = torch.topk(abs_matrix3.flatten(), k)[0].sum()

            # Time-based scaling for each LoRA:
            # Content: strongest at early timesteps (structure)
            # Style: medium throughout
            # Lighting: strongest at late timesteps (refinement)
            scale_content = alpha * (1 - t) + beta
            scale_style = alpha * t + beta
            scale_lighting = self.gamma * t + beta

            if self.pattern == "s*":
                scale_content = scale_content % alpha
                scale_style = scale_style % alpha
                scale_lighting = scale_lighting % self.gamma if self.gamma > 0 else scale_lighting

            # Apply scaling and normalization
            score_content = top_k_sum1 * scale_content / avg_ratio
            score_style = top_k_sum2 * scale_style
            score_lighting = top_k_sum3 * scale_lighting / self.average_ratio_2_3

            # 3-way argmax selection
            scores = torch.tensor([score_content, score_style, score_lighting], device=self.device)
            winner_idx = torch.argmax(scores).item()

            matrices = [matrix1, matrix2, matrix3]
            return matrices[winner_idx]

    def get_klora_weight_2lora(self, timestep):
        """2-LoRA selection (Content + Style only) for comparison mode."""
        sum_timesteps = self.sum_timesteps
        k = self.weight_1_a.shape[1] * self.weight_2_a.shape[1]
        alpha = self.alpha
        beta = self.beta
        avg_ratio = self.average_ratio

        time_ratio = timestep % sum_timesteps

        matrix1 = self.weight_1_a @ self.weight_1_b  # Content
        matrix2 = self.weight_2_a @ self.weight_2_b  # Style

        abs_matrix1 = torch.abs(matrix1)
        top_k_sum1 = torch.topk(abs_matrix1.flatten(), k)[0].sum()

        abs_matrix2 = torch.abs(matrix2)
        top_k_sum2 = torch.topk(abs_matrix2.flatten(), k)[0].sum()

        scale = alpha * time_ratio / sum_timesteps + beta
        if self.pattern == "s*":
            scale = scale % alpha

        top_k_sum1_scaled = top_k_sum1 / avg_ratio
        top_k_sum2_scaled = top_k_sum2 * scale

        temp_ratio = top_k_sum1_scaled / top_k_sum2_scaled
        if temp_ratio > 1:
            return matrix1
        else:
            return matrix2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            glo_count += 1
            weight = self.get_klora_weight(glo_count)
        elif self.forward_type == "weight_1":
            weight = self.weight_1_a @ self.weight_1_b
        elif self.forward_type == "weight_2":
            weight = self.weight_2_a @ self.weight_2_b
        elif self.forward_type == "weight_3" and self.num_loras == 3:
            weight = self.weight_3_a @ self.weight_3_b
        elif self.forward_type == "merge_2lora":
            # Use only content + style (2-LoRA baseline for comparison)
            glo_count += 1
            weight = self.get_klora_weight_2lora(glo_count)
        else:
            raise ValueError(f"Unknown forward_type: {self.forward_type}")
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class KLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
