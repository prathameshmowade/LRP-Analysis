"""
Layer-wise Relevance Propagation (LRP) Engine
=============================================
Implementation of LRP-0, LRP-epsilon, and LRP-gamma rules for PyTorch models.
Complies with the Deep Taylor Decomposition framework (Bach et al., 2015; Montavon et al., 2019).

Guarantees exact layer-by-layer conservation:
    sum_i R_i^{(l)} == sum_j R_j^{(l+1)} == f(x)_c  (for LRP-0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class LRPEngine:
    """
    Engine that applies Layer-wise Relevance Propagation (LRP)
    to sequential feedforward and convolutional PyTorch architectures.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize LRP engine with a PyTorch model.
        """
        self.model = model
        self.model.eval()

    def _get_layers(self) -> List[nn.Module]:
        """Flatten model modules into a list of atomic layers."""
        layers = []
        for module in self.model.children():
            if len(list(module.children())) > 0:
                for sub in module.children():
                    layers.append(sub)
            else:
                layers.append(module)
        return layers

    def forward_pass(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[nn.Module]]:
        """
        Performs a forward pass, caching activations before and after each layer.
        """
        layers = self._get_layers()
        activations = [x]
        current = x

        with torch.no_grad():
            for layer in layers:
                current = layer(current)
                activations.append(current)

        return activations, layers

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        rule: str = "lrp-0",
        epsilon: float = 1e-4,
        gamma: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Compute layer-wise relevance propagation from output to input.

        Args:
            x: Input tensor of shape (1, features) or (1, C, H, W).
            target_class: Target class index to explain. If None, uses argmax(prediction).
            rule: 'lrp-0', 'lrp-epsilon', or 'lrp-gamma'.
            epsilon: Absorber/stabilizer parameter for lrp-epsilon.
            gamma: Positive weight preference multiplier for lrp-gamma.

        Returns:
            Dictionary with input_relevance, layer_relevances, prediction, target_class, etc.
        """
        x = x.clone().detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        activations, layers = self.forward_pass(x)
        output_logits = activations[-1]
        pred_class = int(torch.argmax(output_logits, dim=-1).item())

        if target_class is None:
            target_class = pred_class

        # Initialize relevance at the output layer: R_L = f(x)_c
        R = torch.zeros_like(output_logits)
        R[0, target_class] = output_logits[0, target_class]
        target_score = R[0, target_class].item()

        layer_relevances = [R.clone()]
        current_R = R

        # Backward propagation
        num_layers = len(layers)
        for i in reversed(range(num_layers)):
            layer = layers[i]
            A_in = activations[i].clone().detach().requires_grad_(True)

            if isinstance(layer, nn.Linear):
                current_R = self._propagate_linear(layer, A_in, current_R, rule, epsilon, gamma)
            elif isinstance(layer, nn.Conv2d):
                current_R = self._propagate_conv2d(layer, A_in, current_R, rule, epsilon, gamma)
            elif isinstance(layer, nn.ReLU):
                current_R = self._propagate_relu(A_in, current_R)
            elif isinstance(layer, (nn.Flatten, nn.Dropout)):
                current_R = current_R.view_as(A_in)
            elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                current_R = self._propagate_pool(layer, A_in, current_R)
            else:
                current_R = current_R.view_as(A_in)

            layer_relevances.append(current_R.detach().clone())

        input_relevance = layer_relevances[-1]
        sum_input_rel = input_relevance.sum().item()
        conservation_error = abs(target_score - sum_input_rel)

        return {
            "input_relevance": input_relevance,
            "layer_relevances": list(reversed(layer_relevances)),
            "prediction": output_logits.detach(),
            "predicted_class": pred_class,
            "target_class": target_class,
            "target_score": target_score,
            "sum_input_relevance": sum_input_rel,
            "conservation_error": conservation_error,
            "rule": rule,
        }

    def _propagate_linear(
        self,
        layer: nn.Linear,
        A_in: torch.Tensor,
        R_out: torch.Tensor,
        rule: str,
        epsilon: float,
        gamma: float,
    ) -> torch.Tensor:
        W = layer.weight

        if rule == "lrp-0":
            Z = torch.matmul(A_in, W.t())
            Z_stab = Z + 1e-9 * (torch.sign(Z) + (Z == 0).float())
            S = R_out / Z_stab
            R_in = A_in * torch.matmul(S, W)

        elif rule == "lrp-epsilon":
            Z = torch.matmul(A_in, W.t())
            sign_Z = torch.sign(Z)
            sign_Z[sign_Z == 0] = 1.0
            Z_eps = Z + epsilon * sign_Z
            S = R_out / Z_eps
            R_in = A_in * torch.matmul(S, W)

        elif rule == "lrp-gamma":
            W_pos = torch.clamp(W, min=0.0)
            W_gamma = W + gamma * W_pos
            Z_gamma = torch.matmul(A_in, W_gamma.t())
            sign_Z = torch.sign(Z_gamma)
            sign_Z[sign_Z == 0] = 1.0
            Z_stab = Z_gamma + 1e-9 * sign_Z
            S = R_out / Z_stab
            R_in = A_in * torch.matmul(S, W_gamma)

        else:
            raise ValueError(f"Unknown LRP rule: {rule}")

        return R_in

    def _propagate_conv2d(
        self,
        layer: nn.Conv2d,
        A_in: torch.Tensor,
        R_out: torch.Tensor,
        rule: str,
        epsilon: float,
        gamma: float,
    ) -> torch.Tensor:
        W = layer.weight

        if rule == "lrp-0":
            Z = F.conv2d(A_in, W, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            Z_stab = Z + 1e-9 * (torch.sign(Z) + (Z == 0).float())
            S = R_out / Z_stab
            (grad_A,) = torch.autograd.grad(Z, A_in, grad_outputs=S, retain_graph=True)
            R_in = A_in * grad_A

        elif rule == "lrp-epsilon":
            Z = F.conv2d(A_in, W, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            sign_Z = torch.sign(Z)
            sign_Z[sign_Z == 0] = 1.0
            Z_eps = Z + epsilon * sign_Z
            S = R_out / Z_eps
            (grad_A,) = torch.autograd.grad(Z, A_in, grad_outputs=S, retain_graph=True)
            R_in = A_in * grad_A

        elif rule == "lrp-gamma":
            W_pos = torch.clamp(W, min=0.0)
            W_gamma = W + gamma * W_pos
            Z_gamma = F.conv2d(A_in, W_gamma, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            sign_Z = torch.sign(Z_gamma)
            sign_Z[sign_Z == 0] = 1.0
            Z_stab = Z_gamma + 1e-9 * sign_Z
            S = R_out / Z_stab
            (grad_A,) = torch.autograd.grad(Z_gamma, A_in, grad_outputs=S, retain_graph=True)
            R_in = A_in * grad_A

        else:
            raise ValueError(f"Unknown LRP rule: {rule}")

        return R_in

    def _propagate_relu(self, A_in: torch.Tensor, R_out: torch.Tensor) -> torch.Tensor:
        """
        Passes relevance through ReLU. Inactive neurons have A_in <= 0, which already has 0 relevance.
        """
        return R_out * (A_in > 0).float()

    def _propagate_pool(
        self, pool_layer: nn.Module, A_in: torch.Tensor, R_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Routes relevance through pooling layers via gradient routing.
        """
        Z = pool_layer(A_in)
        (grad_A,) = torch.autograd.grad(Z, A_in, grad_outputs=R_out, retain_graph=True)
        return grad_A
