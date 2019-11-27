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
    %Broca.Layers.MultLayer{params: [x: x, y: y], activation: Broca.Activations.create(activation_type)}
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
      %Broca.Layers.Affine{params: optimize_func.(layer1.params, layer2.grads), activation: layer2.activation}
    end
  end
end

defmodule Broca.Layers.MaxPooling do
  @moduledoc """
  Maximum Pooling Layer which apply filter to find maximum value in it.
  """

  defstruct pool_height: 1,
            pool_width: 1,
            stride: 1,
            padding: 0,
            original_width: 0,
            original_height: 0,
            mask: [],
            activation: nil

  def new(height, width, stride, padding, activation_type \\ nil) do
    %Broca.Layers.MaxPooling{
      pool_height: height,
      pool_width: width,
      stride: stride,
      padding: padding,
      activation: Broca.Activations.create(activation_type)
    }
  end

  defimpl Inspect, for: Broca.Layers.MaxPooling do
    def inspect(pooling, _) do
      "MaxPooling: pool_size= #{pooling.pool_height}x#{pooling.pool_width}, stride=#{
        pooling.stride
      }, padding= #{pooling.padding}, original_size= #{pooling.original_height}x#{
        pooling.original_width
      }, mask=#{Broca.NN.shape_string(pooling.mask)}, activation=#{
        pooling.activation
      }"

    end
  end

  defimpl Layer, for: Broca.Layers.MaxPooling do
    @doc """
    Forward for MaxPooling

    ## Examples
        iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(3, 3, 1, 0)
        iex> Layer.forward(pooling, input)
        {%Broca.Layers.MaxPooling{pool_height: 3, pool_width: 3, stride: 1, padding: 0, original_height: 5, original_width: 5,
         mask: [[[[8, 8, 8], [8, 8, 8], [8, 8, 8]]]]},
          [[[[13, 14, 15], [18, 19, 20], [23, 24, 25]]]]}

        iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]],\
        [[[6, 7, 8, 9, 10], [21, 22, 23, 24, 25], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5], [16, 17, 18, 19, 20]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(3, 3, 1, 0)
        iex> Layer.forward(pooling, input)
        {%Broca.Layers.MaxPooling{pool_height: 3, pool_width: 3, stride: 1, padding: 0, original_height: 5, original_width: 5,
         mask: [[[[8, 8, 8], [8, 8, 8], [8, 8, 8]]], [[[5, 5, 5], [2, 2, 2], [8, 8, 8]]]]},
         [[[[13, 14, 15], [18, 19, 20], [23, 24, 25]]], [[[23, 24, 25], [23, 24, 25], [18, 19, 20]]]]}

        iex> input = [[[[ 63,  72,  81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]], \
        [[[288, 297, 306], [333, 342, 351], [378, 387, 396]], [[576, 594, 612], [666, 684, 702], [756, 774, 792]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(2, 2, 1, 0)
        iex> Layer.forward(pooling, input)
        {%Broca.Layers.MaxPooling{pool_height: 2, pool_width: 2, stride: 1, padding: 0, original_height: 3, original_width: 3,
         mask: [[[[3, 3], [3, 3]], [[3, 3], [3, 3]]], [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]]},
         [[[[117, 126], [162, 171]], [[234, 252], [324, 342]]], [[[342, 351], [387, 396]], [[684, 702], [774, 792]]]]}
    """
    def forward(layer, input) do
      res =
        Broca.NN.matrix_filtering(
          input,
          layer.pool_height,
          layer.pool_width,
          layer.stride,
          layer.padding,
          fn list -> Enum.max(list) end
        )

      {activation, res} =
        if not is_nil(layer.activation) do
          Layer.forward(layer.activation, res)
        else
          {nil, res}
        end

      mask =
        Broca.NN.matrix_filtering(
          input,
          layer.pool_height,
          layer.pool_width,
          layer.stride,
          layer.padding,
          fn list -> Broca.NN.argmax(list) end
        )

      [_, _, height, width] = Broca.NN.shape(input)

      {%Broca.Layers.MaxPooling{
         layer
         | mask: mask,
           original_height: height,
           original_width: width,
           activation: activation
       }, res}
    end

    @doc """
    Backward for MaxPooling

    ## Examples
        iex> pool = %Broca.Layers.MaxPooling{pool_height: 2, pool_width: 2, stride: 1, padding: 0, original_height: 3, original_width: 3, \
        mask: [[[[3, 3], [3, 3]], [[3, 3], [3, 3]]], [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]]}
        iex> Layer.backward(pool, [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], [[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]]]])
        {%Broca.Layers.MaxPooling{pool_height: 2, pool_width: 2, stride: 1, padding: 0, original_height: 3, original_width: 3, \
        mask: [[[[3, 3], [3, 3]], [[3, 3], [3, 3]]], [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]]},
        [[[[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.3, 0.4]],
          [[0.0, 0.0, 0.0], [0.0, 0.5, 0.6], [0.0, 0.7, 0.8]]],
         [[[0.0, 0.0, 0.0], [0.0, 0.9, 1.0], [0.0, 1.1, 1.2]],
          [[0.0, 0.0, 0.0], [0.0, 1.3, 1.4], [0.0, 1.5, 1.6]]]]}
    """
    def backward(layer, dout) do
      {activation, dout} =
        if not is_nil(layer.activation) do
          Layer.backward(layer.activation, dout)
        else
          {nil, dout}
        end

      res =
        Enum.zip(layer.mask, dout)
        |> Enum.map(fn {mask_batch, dout_batch} ->
          Enum.zip(mask_batch, dout_batch)
          |> Enum.map(fn {mask_channel, dout_channel} ->
            {_, result} =
              Enum.zip(mask_channel, dout_channel)
              |> Enum.reduce(
                {0, %{}},
                fn {masks, list}, {idx1, map1} ->
                  {_, result_map} =
                    Enum.zip(masks, list)
                    |> Enum.reduce(
                      {0, map1},
                      fn {mask, val}, {idx2, map} ->
                        idx =
                          (div(mask, layer.pool_width) + idx1) * layer.original_width + idx2 +
                            rem(mask, layer.pool_width)

                        {idx2 + 1, Map.update(map, idx, val, fn v -> v + val end)}
                      end
                    )

                  {idx1 + 1, result_map}
                end
              )

            for y <- 0..(layer.original_height - 1) do
              for x <- 0..(layer.original_width - 1) do
                idx = y * layer.original_width + x
                Map.get(result, idx, 0.0)
              end
            end
          end)
        end)

      {%Broca.Layers.MaxPooling{layer | activation: activation}, res}
    end

    def update(layer, _) do
      %Broca.Layers.MaxPooling{layer | mask: nil}
    end

    def batch_update(_, layer2, _) do
      %Broca.Layers.MaxPooling{layer2 | mask: nil}
    end
  end
end
