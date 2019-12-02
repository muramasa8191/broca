defmodule Broca.Layers.Affine do
  @moduledoc """
  Fully connected layer
  """
  defstruct params: [weight: [], bias: []], x: [], x_shape: [], grads: [], activation: nil

  @doc """
  Constructor

  """
  def new(input_size, output_size, activation_type \\ nil) do
    %Broca.Layers.Affine{
      params: [
        weight:
          Broca.Random.randn(input_size, output_size)
          |> Broca.NN.mult(:math.sqrt(2.0 / input_size)),
        bias: List.duplicate(0.0, output_size)
      ],
      activation: Broca.Activations.create(activation_type)
    }
  end

  defimpl Inspect, for: Broca.Layers.Affine do
    def inspect(affine, _) do
      "Affine: x=#{Broca.NN.shape_string(affine.x)}, x_shape=#{
        Broca.NN.list_string(affine.x_shape)
      }," <>
        "params=[#{
          Keyword.keys(affine.params)
          |> Enum.reduce(
            "",
            fn key, str ->
              ((str <> Atom.to_string(key)) <> ": ") <>
                (affine.params[key] |> Broca.NN.shape_string()) <> ", "
            end
          )
          |> String.trim_trailing(",")
        }]," <>
        "grads=[#{
          Keyword.keys(affine.grads)
          |> Enum.reduce(
            "",
            fn key, str ->
              str <>
                Atom.to_string(key) <>
                ": " <>
                (affine.grads[key] |> Broca.NN.shape_string()) <> ", "
            end
          )
          |> String.trim_trailing(",")
        }]," <>
        "activation=#{affine.activation}"
    end
  end

  defimpl Layer, for: Broca.Layers.Affine do
    @doc """

    ## Examples
        iex> layer = %Broca.Layers.Affine{params: [weight: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], bias: [0.9, 0.6, 0.3]]}
        iex> Layer.forward(layer, [-0.12690894,  0.31470161])
        {%Broca.Layers.Affine{params: [weight: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], bias: [0.9, 0.6, 0.3]], x: [-0.12690894,  0.31470161], x_shape: [2]}, [1.01318975, 0.7319690169999999, 0.450748284]}
    """
    def forward(layer, x) do
      shape = Broca.NN.shape(x)

      x =
        if length(shape) > 2 do
          Broca.NN.reshape(x, [hd(shape), tl(shape) |> Enum.reduce(1, fn x, acc -> acc * x end)])
        else
          x
        end

      out = Broca.NN.dot(x, layer.params[:weight]) |> Broca.NN.add(layer.params[:bias])

      {activation, out} =
        if not is_nil(layer.activation) do
          Layer.forward(layer.activation, out)
        else
          {nil, out}
        end

      {%Broca.Layers.Affine{layer | x: x, x_shape: shape, activation: activation}, out}
    end

    @doc """
    Backward for Affine Layer

    ## Examples
        iex> dout = [[0.84029972, 0.22303644, 0.19933264]]
        iex> layer = %Broca.Layers.Affine{params: [weight: [[0.4176742 , 2.3945292 , -0.36417824],[0.35595952, 0.31028157, -0.84189487]], \
                     bias: [0.55537918, -0.1362438 , -1.3922125]], x: [0.96726788, 0.09582064], x_shape: [2]}
        iex> Layer.backward(layer, dout)
        {%Broca.Layers.Affine{params: [weight: [[0.4176742 ,  2.3945292 , -0.36417824],[0.35595952,
        0.31028157, -0.84189487]], bias: [0.55537918, -0.1362438 , -1.3922125]], x: [0.96726788, 0.09582064], x_shape: [2],
        grads: [weight: [[0.8127949287289935, 0.21573598448154718, 0.1928080601076032],[0.0805180569622208, 0.0213714944241216, 0.0191001811376896]], 
        bias: [0.84029972, 0.22303644, 0.19933264]]},[0.8124461715455185, 0.20049965471818834]}
    """
    def backward(layer, dout) do
      {activation, dout} =
        if not is_nil(layer.activation) do
          Layer.backward(layer.activation, dout)
        else
          {nil, dout}
        end

      dx = Broca.NN.dot(dout, Broca.NN.transpose(layer.params[:weight]))
      dw = Broca.NN.transpose(layer.x) |> Broca.NN.dot(dout)
      db = if is_list(hd(dout)), do: Broca.NN.sum(dout, :col), else: dout

      {%Broca.Layers.Affine{layer | grads: [weight: dw, bias: db], activation: activation},
       Broca.NN.reshape(dx, layer.x_shape)}
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

      %Broca.Layers.Affine{params: updated_params, activation: layer.activation}
    end

    def batch_update(layer1, layer2, optimize_func) do
      %Broca.Layers.Affine{
        params: optimize_func.(layer1.params, layer2.grads),
        activation: layer2.activation
      }
    end
  end
end
