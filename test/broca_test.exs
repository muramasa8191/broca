defmodule BrocaTest do
  use ExUnit.Case

  doctest Broca
  doctest Broca.NN
  doctest Broca.Random

  doctest Layer.Broca.Activations.ReLU
  doctest Layer.Broca.Activations.Sigmoid
  doctest Layer.Broca.Activations.Softmax

  doctest Layer.Broca.Layers.Affine
  doctest Layer.Broca.Layers.MultLayer
  doctest Layer.Broca.Layers.MaxPooling
  doctest Layer.Broca.Layers.Convolution
  doctest Layer.Broca.Layers.Dropout

  doctest Loss.Broca.Losses.CrossEntropyError
  doctest Loss.Broca.Losses.SoftmaxWithLoss

  doctest Broca.Optimizers.AdaGrad
  doctest Broca.Layers.Convolution
  doctest Broca.Layers.Dropout
  doctest Broca.Optimizer.Broca.Optimizers.AdaGrad
end
