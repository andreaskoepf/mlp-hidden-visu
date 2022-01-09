# MLP Hidden Layer Activation Visualization

To gain some intuition about the internal representation of simple multi-layer perceptrons (MLPs) I trained a neural network with [PyTorch](https://pytorch.org/) using a range of different activation functions on a 2D -> 1D function. The training pairs consist of the u,v image coordinates [0, 1]^2 as inputs (first row) and the intensities from a 4x4 checkerboard pattern as targets (last row). The other rows show intensities of neurons in the hidden layers for all the u,v input coordinates (one box=one neuron). The animations show how the neuron responses change over the course of the first 4000 steps trained with Adam (lr=0.01, loss_fn=MSE). For further details please refer to the implementation in [main.py](./main.py).

## Sigmoid
![sigmoid](results/sigmoid_animation.gif)

---

## tanh
[tanh](https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh)
![tanh](results/tanh_animation.gif)

---
## ReLU
[ReLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu)
![relu](results/relu_animation.gif)

---

## LeakyReLU
[LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html#torch.nn.functional.leaky_relu)
![leaky_relu](results/leakyrelu_animation.gif)

---

## ELU
[ELU](https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html#torch.nn.functional.elu)
![elu](results/elu_animation.gif)

---

## Cosine
![cos](results/cos_animation.gif)

---

## CELU
[CELU](https://pytorch.org/docs/stable/generated/torch.nn.functional.celu.html#torch.nn.functional.celu)
![celu](results/celu_animation.gif)

---

## GELU
[GELU](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html#torch.nn.functional.gelu)
![gelu](results/gelu_animation.gif)

---

## Mish
[Mish](https://pytorch.org/docs/stable/generated/torch.nn.functional.mish.html#torch.nn.functional.mish)
![mish](results/mish_animation.gif)

---

## SELU
[SELU](https://pytorch.org/docs/stable/generated/torch.nn.functional.selu.html#torch.nn.functional.selu)
![selu](results/selu_animation.gif)

---

## SiLU
[SiLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html#torch.nn.functional.silu)
![silu](results/silu_animation.gif)
