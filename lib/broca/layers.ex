defprotocol Layer do
  def forward(layer, input)
  def backward(layer, dout)
  def update(layer, optimize_func)
  def batch_update(layer1, layer2, optimize_func)
end

defmodule Broca.Layers.MultLayer do
  @moduledoc """
  Multiplication Layer
  """
  defstruct params: [], grads: [], activation: nil

  def new(x, y, activation_type \\ nil) do
    %Broca.Layers.MultLayer{
      params: [x: x, y: y],
      activation: Broca.Activations.create(activation_type)
    }
  end

  defimpl Layer, for: Broca.Layers.MultLayer do
    @doc """
    Forward

    ## Examples
        iex> Layer.forward(%Broca.Layers.MultLayer{}, [1, 2])
        {%Broca.Layers.MultLayer{params: [x: 1, y: 2]}, 2}

        iex> Layer.forward(%Broca.Layers.MultLayer{}, [[1, 2], [3, 4]])
        {%Broca.Layers.MultLayer{params: [x: [1, 2], y: [3, 4]]}, [3, 8]}
    """
    def forward(_, [x, y]) do
      {Broca.Layers.MultLayer.new(x, y), Broca.NN.mult(x, y)}
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Layers.MultLayer{params: [x: 1, y: 2]}, 2.0)
        {%Broca.Layers.MultLayer{params: [x: 1, y: 2]}, [4.0, 2.0]}
    """
    def backward(layer, dout) do
      {layer, [Broca.NN.mult(layer.params[:y], dout), Broca.NN.mult(layer.params[:x], dout)]}
    end

    @doc """
    Implementation of Layer but do nothing.
    """
    def update(layer, _) do
      layer
    end

    def batch_update(layer, _, _) do
      layer
    end
  end
end
