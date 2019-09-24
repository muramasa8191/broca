defmodule BrocaTest do
  use ExUnit.Case

  doctest Broca
  doctest Broca.NN

  doctest Layer.Broca.Activations.ReLU
  doctest Layer.Broca.Activations.Sigmoid
  doctest Layer.Broca.Activations.Softmax

  doctest Layer.Broca.Layers.Affine
  doctest Layer.Broca.Layers.MultLayer

  doctest Loss.Broca.Losses.CrossEntropyError
  doctest Loss.Broca.Losses.SoftmaxWithLoss

  doctest Broca.Optimizers.AdaGrad
  doctest Optimizer.Broca.Optimizers.AdaGrad
end
