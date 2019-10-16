defprotocol Layer do
  def forward(layer, input)
  def backward(layer, dout)
  def update(layer, optimize_func)
  def batch_update(layer1, layer2)
  def get_grads(layer)
  def gradient_forward(layer, x, name, idx1, idx2, idx3, diff)
end

defmodule Broca.Layers.MultLayer do
  @moduledoc """
  Multiplication Layer
  """
  defstruct params: [], grads: []

  def new(x, y) do
    %Broca.Layers.MultLayer{params: [x: x, y: y]}
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

    def gradient_forward(layer, x, _, _, _, _, _) do
      forward(layer, x)
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Layers.MultLayer{params: [x: 1, y: 2]}, 2.0)
        {%Broca.Layers.MultLayer{params: [x: 1, y: 2]}, [4.0, 2.0]}
    """
    def backward(layer, dout) do
      {layer, [dout * layer.params[:y], dout * layer.params[:x]]}
    end

    @doc """
    Implementation of Layer but do nothing.
    """
    def update(layer, _) do
      layer
    end

    def batch_update(layer, _) do
      layer
    end

    def get_grads(_) do
      []
    end
  end
end

defmodule Broca.Layers.Affine do
  @moduledoc """
  Fully connected layer
  """
  defstruct name: "affine", params: [weight: [], bias: []], x: [], grads: nil

  @doc """
  Constructor

  ## Examples
      iex> Broca.Layers.Affine([0.1, 0.2], [0.3, 0.4])
      %Broca.Layers.Affine{weight: [0.1, 0.2], bias: [0.3, 0.4]}
  """
  def new(name, w, b) do
    %Broca.Layers.Affine{name: name, params: [weight: w, bias: b]}
  end

  defimpl Layer, for: Broca.Layers.Affine do
    @doc """

    ## Examples
        iex> layer = Broca.Layers.Affine.new("a1", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0.9, 0.6, 0.3])
        iex> Layer.forward(layer, [-0.12690894,  0.31470161])
        {%Broca.Layers.Affine{name: "a1", params: [weight: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], bias: [0.9, 0.6, 0.3]], x: [[-0.12690894,  0.31470161]]}, [[1.01318975, 0.7319690169999999, 0.450748284]]}
    """
    def forward(layer, x) do
      x = if is_list(hd(x)), do: x, else: [x]
      out = Broca.NN.dot(x, layer.params[:weight]) |> Broca.NN.add(layer.params[:bias])
      {%Broca.Layers.Affine{layer | x: x}, out}
    end

    def gradient_forward(layer, x, name, idx1, idx2, idx3, diff) do
      out =
        if name == layer.name do
          Broca.NumericalGradient.dot2(x, layer.params[:weight], idx1, idx2, diff)
          |> Broca.NumericalGradient.add2(layer.params[:bias], idx3, diff)
        else
          {_, res} = forward(layer, x)
          res
        end

      {%Broca.Layers.Affine{layer | x: x}, out}
    end

    @doc """
    Backward for Affine Layer

    ## Examples
        iex> dout = [[0.84029972, 0.22303644, 0.19933264]]
        iex> layer = %Broca.Layers.Affine{params: [weight: [[ 0.4176742 , 2.3945292 , -0.36417824],[0.35595952, 0.31028157, -0.84189487]], \
                     bias: [0.55537918, -0.1362438 , -1.3922125]], x: [[0.96726788, 0.09582064]]}
        iex> Layer.backward(layer, dout)
        {%Broca.Layers.Affine{params: [weight: [[0.4176742 ,  2.3945292 , -0.36417824],[0.35595952,
        0.31028157, -0.84189487]], bias: [0.55537918, -0.1362438 , -1.3922125]], x: [[0.96726788, 0.09582064]],
        grads: [weight: [[0.8127949287289935, 0.21573598448154718, 0.1928080601076032],[0.0805180569622208, 0.0213714944241216, 0.0191001811376896]], 
        bias: [0.84029972, 0.22303644, 0.19933264]]},[[0.8124461715455185, 0.20049965471818834]]}
    """
    def backward(layer, dout) do
      dout_dot = if is_list(hd(dout)), do: dout, else: [dout]
      dx = Broca.NN.dot(dout_dot, Broca.NN.transpose(layer.params[:weight]))
      dw = Broca.NN.transpose(layer.x) |> Broca.NN.dot(dout_dot)
      db = if is_list(hd(dout)), do: Broca.NN.sum(dout, :col), else: dout
      {%Broca.Layers.Affine{layer | grads: [weight: dw, bias: db]}, dx}
    end

    @doc """
    Implementation of Layer but do nothing.
    """
    def update(layer, optimize_func) do
      updated_params =
        Keyword.keys(layer.params)
        |> Enum.reduce([], fn key, list ->
          Keyword.put_new(list, key, optimize_func.(layer.params[key], layer.grads[key]))
        end)

      %Broca.Layers.Affine{name: layer.name, params: updated_params}
    end

    def batch_update(%Broca.Layers.Affine{grads: g}, layer2) when is_nil(g) do
      %Broca.Layers.Affine{
        params: layer2.params,
        grads: layer2.grads
      }
    end

    def batch_update(layer1, layer2) do
      dw = Broca.NN.add(layer1.grads[:weight], layer2.grads[:weight])
      db = Broca.NN.add(layer1.grads[:bias], layer2.grads[:bias])

      %Broca.Layers.Affine{
        params: [weight: layer1.params[:weight], bias: layer1.params[:bias]],
        grads: [weight: dw, bias: db]
      }
    end

    def get_grads(layer) do
      [layer.grads[:weight], layer.grads[:bias]]
    end
  end
end

defmodule Broca.Layers.MaxPooling do
  @moduledoc """
  Maximum Pooling Layer which apply filter to find maximum value in it.
  """

  defstruct pool_height: 1, pool_width: 1, stride: 1, padding: 0

  def new(height, width, stride, padding) do
    %Broca.Layers.MaxPooling{
      pool_height: height,
      pool_width: width,
      stride: stride,
      padding: padding
    }
  end

  defimpl Layer, for: Broca.Layers.MaxPooling do
    @doc """
    Forward for MaxPooling

    ## Examples
        iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(3, 3, 1, 0)
        iex> Layer.forward(pooling, input)
        [[[[13, 14, 15], [18, 19, 20], [23, 24, 25]]]]

        iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]], [[[6, 7, 8, 9, 10], [21, 22, 23, 24, 25], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5], [16, 17, 18, 19, 20]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(3, 3, 1, 0)
        iex> Layer.forward(pooling, input)
        [[[[13, 14, 15], [18, 19, 20], [23, 24, 25]]], [[[23, 24, 25], [23, 24, 25], [18, 19, 20]]]]
    """
    def forward(layer, input) do
      Broca.NN.matrix_filtering(
        input,
        layer.pool_height,
        layer.pool_width,
        layer.stride,
        layer.padding,
        fn list -> Enum.max(list) end
      )
    end

    def backward(layer, dout) do
    end

    def update(layer, optimize_func) do
    end

    def batch_update(layer1, layer2) do
    end

    def get_grads(layer) do
    end

    def gradient_forward(layer, x, name, idx1, idx2, idx3, diff) do
    end
  end
end
