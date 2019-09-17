defprotocol Layer do
  def forward(layer, input)
  def backward(layer, dout)
end

defmodule Model do
  @type t :: Layer.t()

  @spec forward([t], [number]) :: [t]
  def forward(layers, input) do
    {outs, _} =
      layers |> Enum.reduce({[], input},
                              fn layer, {layers, input} ->
                                {layer, out} = Layer.forward(layer, input)
                                {[layer]++layers, out}
                              end)
    Enum.reverse(outs)
  end
  @spec backward([t], [number]) :: [t]
  def backward(layers, dout) do
    {outs, _} =
      Enum.reverse(layers)
       |> Enum.reduce({[], dout},
                      fn layer, {layers, dout} ->
                        {layer, out} = Layer.backward(layer, dout)
                        {[layer]++layers, out}
                      end)
    outs
  end
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
      iex> Model.forward([%Broca.Layers.MultLayer{}], [1, 2])
      [%Broca.Layers.MultLayer{x: 1, y: 2}]

      iex> Model.forward([%Broca.Layers.MultLayer{}], [[1, 2], [3, 4]])
      [%Broca.Layers.MultLayer{x: [1, 2], y: [3, 4]}]
    """
    def forward(_, [x, y]) do
      {Broca.Layers.MultLayer.new(x, y), Broca.NN.mult(x, y)}
    end

    @doc """
    Backward

    ## Examples
      iex> Model.backward([%Broca.Layers.MultLayer{x: 1, y: 2}], 2.0)
      [%Broca.Layers.MultLayer{x: 1, y: 2}]
    """
    def backward(layer, dout) do
      {layer, [dout * layer.y, dout * layer.x]}
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
      iex> layer = Broca.Layers.Affine.new([[0.1, 0.2, 0.3], [0.4, 0.5, 0,6]], [0.9, 0.6, 0.3])
      iex> Model.forward([layer], [-0.12690894,  0.31470161])
      [%Broca.Layers.Affine{weight: [[0.1, 0.2, 0.3], [0.4, 0.5, 0,6]], bias: [0.9, 0.6, 0.3], x: [-0.12690894,  0.31470161]}]
    """
    def forward(layer, x) do
      out = x |> Broca.NN.dot(layer.weight) |> Broca.NN.add(layer.bias)
      {%Broca.Layers.Affine{layer | x: x}, out}
    end
    def backward(layer, dout) do
      dx = dout |> Broca.NN.dot(Broca.NN.transpose(layer.weight))
      dw = Broca.NN.transpose(layer.x) |> Broca.NN.dot(dout)
      db = Broca.NN.sum(dout, :col)
      {%Broca.Layers.Affine{layer| dw: dw, db: db}, dx}
    end
  end
end