defmodule Broca.Models do
  alias Broca.{
    Layers
  }

  def load_cifar_conv_net do
    [
      Layers.convolution2d(3, 3, 3, 32, 1, 1),
      Layers.relu(),
      Layers.convolution2d(3, 3, 32, 32, 1, 1),
      Layers.relu(),
      Layers.max_pooling2d(2, 2, 2, 0),
      Layers.dropout(0.25),
      Layers.convolution2d(3, 3, 32, 64, 1, 1),
      Layers.relu(),
      Layers.convolution2d(3, 3, 64, 64, 1, 1),
      Layers.relu(),
      Layers.max_pooling2d(2, 2, 2, 0),
      Layers.dropout(0.25),
      Layers.affine(4096, 512),
      Layers.relu(),
      Layers.dropout(0.5),
      Layers.affine(512, 10),
      Layers.softmax()
    ]
  end

  def load_mnist_conv_net do
    [
      Layers.convolution2d(5, 5, 1, 30, 1, 0),
      Layers.relu(),
      Layers.max_pooling2d(2, 2, 2, 0),
      Layers.affine(4320, 100),
      Layers.relu(),
      Layers.affine(100, 10),
      Layers.softmax()
    ]
  end

  def load_mnist_two_layer_net do
    [
      Layers.affine(784, 100),
      Layers.relu(),
      Layers.affine(100, 10),
      Layers.softmax()
    ]
  end
end
