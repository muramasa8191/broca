defprotocol Layer do
  def forward(layer, input)
  def backward(layer, dout)
  def update(layer, learning_rate)
end

defmodule Broca.Layers.MultLayer do
  defstruct x: nil, y: nil

  def new(x, y) do
    %Broca.Layers.MultLayer{x: x, y: y}
  end

  defimpl Layer, for: Broca.Layers.MultLayer do
    @doc """
    Forward

    ## Examples
        iex> Layer.forward(%Broca.Layers.MultLayer{}, [1, 2])
        {%Broca.Layers.MultLayer{x: 1, y: 2}, 2}

        iex> Layer.forward(%Broca.Layers.MultLayer{}, [[1, 2], [3, 4]])
        {%Broca.Layers.MultLayer{x: [1, 2], y: [3, 4]}, [3, 8]}
    """
    def forward(_, [x, y]) do
      {Broca.Layers.MultLayer.new(x, y), Broca.NN.mult(x, y)}
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Layers.MultLayer{x: 1, y: 2}, 2.0)
        {%Broca.Layers.MultLayer{x: 1, y: 2}, [4.0, 2.0]}
    """
    def backward(layer, dout) do
      {layer, [dout * layer.y, dout * layer.x]}
    end

    @doc """
    Implementation of Layer but do nothing.
    """
    def update(layer, _) do
      layer
    end
  end
end

defmodule Broca.Layers.Affine do
  @moduledoc """
  Fully connected layer
  """
  defstruct weight: [], bias: [], x: [], dw: [], db: []

  @doc """
  Constructor

  ## Examples
      iex> Broca.Layers.Affine([0.1, 0.2], [0.3, 0.4])
      %Broca.Layers.Affine{weight: [0.1, 0.2], bias: [0.3, 0.4]}
  """
  def new(w, b) do
    %Broca.Layers.Affine{weight: w, bias: b}
  end

  defimpl Layer, for: Broca.Layers.Affine do
    @doc """

    ## Examples
        iex> layer = Broca.Layers.Affine.new([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0.9, 0.6, 0.3])
        iex> Layer.forward(layer, [-0.12690894,  0.31470161])
        {%Broca.Layers.Affine{weight: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], bias: [0.9, 0.6, 0.3], x: [-0.12690894,  0.31470161]}, [1.01318975, 0.7319690169999999, 0.450748284]}
    """
    def forward(layer, x) do
      out = x |> Broca.NN.dot(layer.weight) |> Broca.NN.add(layer.bias)
      {%Broca.Layers.Affine{layer | x: x}, out}
    end

    @doc """
    Backward for Affine Layer

    ## Examples
        iex> dout = [[0.84029972, 0.22303644, 0.19933264]]
        iex> layer = %Broca.Layers.Affine{weight: [[ 0.4176742 , 2.3945292 , -0.36417824],[0.35595952,\
             0.31028157, -0.84189487]], bias: [0.55537918, -0.1362438 , -1.3922125], x: [[0.96726788, 0.09582064]]}
        iex> Layer.backward(layer, dout)
        {%Broca.Layers.Affine{weight: [[0.4176742 ,  2.3945292 , -0.36417824],[0.35595952,
        0.31028157, -0.84189487]], bias: [0.55537918, -0.1362438 , -1.3922125], x: [[0.96726788, 0.09582064]],
        dw: [[0.8127949287289935, 0.21573598448154718, 0.1928080601076032],[0.0805180569622208, 0.0213714944241216, 0.0191001811376896]], 
        db: [0.84029972, 0.22303644, 0.19933264]},[[0.8124461715455185, 0.20049965471818834]]}
    """
    def backward(layer, dout) do
      dx = dout |> Broca.NN.dot(Broca.NN.transpose(layer.weight))
      dw = Broca.NN.transpose(layer.x) |> Broca.NN.dot(dout)
      db = Broca.NN.sum(dout, :col)
      {%Broca.Layers.Affine{layer | dw: dw, db: db}, dx}
    end

    @doc """
    Implementation of Layer but do nothing.
    """
    def update(layer, learning_rate) do
      %Broca.Layers.Affine{
        weight: Broca.NN.subtract(layer.weight, Broca.NN.mult(layer.dw, learning_rate)),
        bias: Broca.NN.subtract(layer.bias, Broca.NN.mult(layer.db, learning_rate))
      }
    end
  end
end
