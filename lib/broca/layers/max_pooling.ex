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
      }, mask=#{Broca.NN.shape_string(pooling.mask)}, activation=#{pooling.activation}"
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

        iex> input = [[[[63, 72, 81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]], \
        [[[288, 297, 306], [333, 342, 351], [378, 387, 396]], [[576, 594, 612], [666, 684, 702], [756, 774, 792]]]]
        iex> pooling = Broca.Layers.MaxPooling.new(2, 2, 1, 0)
        iex> Layer.forward(pooling, input)
        {%Broca.Layers.MaxPooling{pool_height: 2, pool_width: 2, stride: 1, padding: 0, original_height: 3, original_width: 3,
         mask: [[[[3, 3], [3, 3]], [[3, 3], [3, 3]]], [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]]},
         [[[[117, 126], [162, 171]], [[234, 252], [324, 342]]], [[[342, 351], [387, 396]], [[684, 702], [774, 792]]]]}
    """
    def forward(layer, input) do
      # s = System.os_time(:millisecond)

      res =
        Broca.NN.matrix_filtering(
          input,
          layer.pool_height,
          layer.pool_width,
          layer.stride,
          layer.padding,
          fn list -> Enum.max(list) end,
          :normal
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
          fn list -> Broca.NN.argmax(list) end,
          :normal
        )

      [_, _, height, width] = Broca.NN.shape(input)

      # IO.puts("** MaxPooling forward: #{System.os_time(:millisecond) - s}msec")

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

      # s = System.os_time(:millisecond)

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

      # IO.puts("** MaxPooling backward: #{System.os_time(:millisecond) - s}msec")

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
