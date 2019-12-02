defmodule Broca.Layers.Dropout do
  defstruct mask: nil, ratio: 0.25

  def new(ratio) do
    %Broca.Layers.Dropout{ratio: ratio}
  end

  @doc """
  Create the mask for the shape given.
  If the random number is less than ratio, the grid would be masked.

  """
  def create_mask([n, c, h, w], ratio) do
    for _ <- 1..n do
      for _ <- 1..c do
        for _ <- 1..h do
          for _ <- 1..w do
            :rand.uniform() < ratio
          end
        end
      end
    end
  end

  def create_mask([h, w], ratio) do
    for _ <- 1..h do
      for _ <- 1..w do
        :rand.uniform() < ratio
      end
    end
  end

  defimpl Layer, for: Broca.Layers.Dropout do
    def forward(layer, input) do
      mask = Broca.Layers.Dropout.create_mask(Broca.NN.shape(input), layer.ratio)
      {%Broca.Layers.Dropout{layer | mask: mask}, Broca.NN.mask(mask, input, 0.0)}
    end

    def backward(layer, dout) do
      {layer, Broca.NN.mask(layer.mask, dout, 0.0)}
    end

    def update(layer, _) do
      %Broca.Layers.Dropout{layer | mask: nil}
    end

    def batch_update(_, layer, _) do
      %Broca.Layers.Dropout{layer | mask: nil}
    end
  end
end
