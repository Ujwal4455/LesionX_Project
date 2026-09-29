import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def _hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor):
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
        self._hook()
        logits = self.model(image_tensor)
        score = logits[:, logits.argmax(dim=1)].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        return cam
