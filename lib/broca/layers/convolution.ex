defmodule Broca.Layers.Convolution do
  @moduledoc """
  Convolution Layer
  """

  defstruct filter_height: 1,
            filter_width: 1,
            stride: 1,
            padding: 0,
            params: [],
            grads: [],
            col: nil,
            activation: nil

  @doc """
  Helper function to create Convolution
  """
  def new(
        height,
        width,
        input_channel_size,
        filter_size,
        weight_init_std \\ 0.01,
        activation_type \\ nil
      ) do
    %Broca.Layers.Convolution{
      filter_height: height,
      filter_width: width,
      params: [
        weight:
          Broca.Random.randn(filter_size, input_channel_size * width * height)
          |> Broca.NN.mult(weight_init_std),
        bias: List.duplicate(0.0, filter_size)
      ],
      activation: Broca.Activations.create(activation_type)
    }
  end

  defimpl Inspect, for: Broca.Layers.Convolution do
    def inspect(conv, _) do
      "Convolution: filter size=#{conv.filter_height}x#{conv.filter_width}, stride=#{conv.stride}, padding=#{
        conv.padding
      }, " <>
        "params=[#{
          Keyword.keys(conv.params)
          |> Enum.reduce(
            "",
            fn key, str ->
              ((str <> Atom.to_string(key)) <> ": ") <>
                (conv.params[key] |> Broca.NN.shape_string()) <> ", "
            end
          )
          |> String.trim_trailing(", ")
        }]," <>
        "grads=[#{
          Keyword.keys(conv.grads)
          |> Enum.reduce(
            "",
            fn key, str ->
              str <>
                Atom.to_string(key) <>
                ": [" <>
                (conv.grads[key] |> Broca.NN.shape_string()) <> ", "
            end
          )
          |> String.trim_trailing(",")
        }], " <>
        "col=#{Broca.NN.shape_string(conv.col)}, activation=#{conv.activation}"
    end
  end

  defimpl Layer, for: Broca.Layers.Convolution do
    @doc """
    Forward for Convolution

    ## Examples
      iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]], \
      [[[26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40], [41, 42, 43, 44, 45], [46, 47, 48, 49, 50]]]]
      iex> conv = %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [0, 0]]}
      iex> Layer.forward(conv, input)
      {%Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [0, 0]],
        col: [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [2, 3, 4, 7, 8, 9, 12, 13, 14], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[6, 7, 8, 11, 12, 13, 16, 17, 18], [7, 8, 9, 12, 13, 14, 17, 18, 19], [8, 9, 10, 13, 14, 15, 18, 19, 20]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [12, 13, 14, 17, 18, 19, 22, 23, 24], [13, 14, 15, 18, 19, 20, 23, 24, 25]]],
      [[[26, 27, 28, 31, 32, 33, 36, 37, 38], [27, 28, 29, 32, 33, 34, 37, 38, 39], [28, 29, 30, 33, 34, 35, 38, 39, 40]],
       [[31, 32, 33, 36, 37, 38, 41, 42, 43], [32, 33, 34, 37, 38, 39, 42, 43, 44], [33, 34, 35, 38, 39, 40, 43, 44, 45]],
       [[36, 37, 38, 41, 42, 43, 46, 47, 48], [37, 38, 39, 42, 43, 44, 47, 48, 49], [38, 39, 40, 43, 44, 45, 48, 49, 50]]]]},
      [[[[ 63,  72,  81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]],
       [[[288, 297, 306], [333, 342, 351], [378, 387, 396]], [[576, 594, 612], [666, 684, 702], [756, 774, 792]]]]}

      iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]], \
      [[[26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40], [41, 42, 43, 44, 45], [46, 47, 48, 49, 50]]]]
      iex> conv = %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [1, 2]]}
      iex> Layer.forward(conv, input)
      {%Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [1, 2]],
        col: [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [2, 3, 4, 7, 8, 9, 12, 13, 14], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[6, 7, 8, 11, 12, 13, 16, 17, 18], [7, 8, 9, 12, 13, 14, 17, 18, 19], [8, 9, 10, 13, 14, 15, 18, 19, 20]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [12, 13, 14, 17, 18, 19, 22, 23, 24], [13, 14, 15, 18, 19, 20, 23, 24, 25]]],
      [[[26, 27, 28, 31, 32, 33, 36, 37, 38], [27, 28, 29, 32, 33, 34, 37, 38, 39], [28, 29, 30, 33, 34, 35, 38, 39, 40]],
       [[31, 32, 33, 36, 37, 38, 41, 42, 43], [32, 33, 34, 37, 38, 39, 42, 43, 44], [33, 34, 35, 38, 39, 40, 43, 44, 45]],
       [[36, 37, 38, 41, 42, 43, 46, 47, 48], [37, 38, 39, 42, 43, 44, 47, 48, 49], [38, 39, 40, 43, 44, 45, 48, 49, 50]]]]},
      [[[[ 64,  73,  82], [109, 118, 127], [154, 163, 172]], [[128, 146, 164], [218, 236, 254], [308, 326, 344]]],
       [[[289, 298, 307], [334, 343, 352], [379, 388, 397]], [[578, 596, 614], [668, 686, 704], [758, 776, 794]]]]}
    """
    def forward(layer, input) do
      col =
        Broca.NN.matrix_filtering(
          input,
          layer.filter_height,
          layer.filter_width,
          layer.stride,
          layer.padding
        )

      out =
        col
        |> Enum.map(fn batch ->
          Enum.map(batch, fn channel ->
            Enum.map(channel, fn data ->
              data
              |> Broca.NN.mult(layer.params[:weight])
              |> Broca.NN.sum(:row)
              |> Broca.NN.add(layer.params[:bias])
            end)
          end)
        end)

      [batch_size, _, height, width] = Broca.NN.shape(input)
      out_height = 1 + div(height + 2 * layer.padding - layer.filter_height, layer.stride)
      out_width = 1 + div(width + 2 * layer.padding - layer.filter_width, layer.stride)
      total = Broca.NN.shape(out) |> Enum.reduce(1, &(&1 * &2))

      res =
        out
        |> Broca.NN.reshape([
          batch_size,
          out_height,
          out_width,
          div(total, batch_size * out_height * out_width)
        ])
        |> Broca.NN.transpose(0, 3, 1, 2)

      {activation, res} =
        if not is_nil(layer.activation) do
          Layer.forward(layer.activation, res)
        else
          {nil, res}
        end

      {%Broca.Layers.Convolution{layer | col: col, activation: activation}, res}
    end

    @doc """
    Backward for Convolution

    ## Examples
        iex> dout = [[[[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.3, 0.4]], [[0.0, 0.0, 0.0], [0.0, 0.5, 0.6], [0.0, 0.7, 0.8]]], \
        [[[0.0, 0.0, 0.0], [0.0, 0.9, 1.0], [0.0, 1.1, 1.2]], [[0.0, 0.0, 0.0], [0.0, 1.3, 1.4], [0.0, 1.5, 1.6]]]]
        iex> conv = %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [0, 0]], \
        col: [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [2, 3, 4, 7, 8, 9, 12, 13, 14], [3, 4, 5, 8, 9, 10, 13, 14, 15]], \
       [[6, 7, 8, 11, 12, 13, 16, 17, 18], [7, 8, 9, 12, 13, 14, 17, 18, 19], [8, 9, 10, 13, 14, 15, 18, 19, 20]], \
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [12, 13, 14, 17, 18, 19, 22, 23, 24], [13, 14, 15, 18, 19, 20, 23, 24, 25]]], \
      [[[26, 27, 28, 31, 32, 33, 36, 37, 38], [27, 28, 29, 32, 33, 34, 37, 38, 39], [28, 29, 30, 33, 34, 35, 38, 39, 40]], \
       [[31, 32, 33, 36, 37, 38, 41, 42, 43], [32, 33, 34, 37, 38, 39, 42, 43, 44], [33, 34, 35, 38, 39, 40, 43, 44, 45]], \
       [[36, 37, 38, 41, 42, 43, 46, 47, 48], [37, 38, 39, 42, 43, 44, 47, 48, 49], [38, 39, 40, 43, 44, 45, 48, 49, 50]]]]}
        iex> Layer.backward(conv, dout)
        {
          %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [0, 0]], 
          grads: [weight: [[159.20000000000002, 164.4, 169.6, 185.20000000000002, 190.4, 195.6, 211.20000000000002, 216.4, 221.6], [231.20000000000002, 239.6, 248.0, 273.20000000000005, 281.6, 290.0, 315.20000000000005, 323.59999999999997, 332.0]],
          bias: [5.199999999999999, 8.4]]},
          [[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.1, 2.5, 2.5, 1.4], [0.0, 2.8, 6.199999999999999, 6.199999999999999, 3.4000000000000004],
          [0.0, 2.8, 6.199999999999999, 6.199999999999999, 3.4000000000000004], [0.0, 1.7, 3.6999999999999997, 3.6999999999999997, 2.0]]],
          [[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.5, 7.3, 7.3, 3.8], [0.0, 7.6, 15.8, 15.8, 8.2], [0.0, 7.6, 15.8, 15.8, 8.2], [0.0, 4.1, 8.5, 8.5, 4.4]]]]
        }
    """
    def backward(layer, dout) do
      {activation, dout} =
        if not is_nil(layer.activation) do
          Layer.backward(layer.activation, dout)
        else
          {nil, dout}
        end

      db =
        dout
        |> Enum.reduce(
          List.duplicate(0.0, length(hd(dout))),
          fn batch, acc ->
            Enum.map(batch, &(Broca.NN.sum(&1) |> Enum.sum()))
            |> Broca.NN.add(acc)
          end
        )

      dw =
        Enum.zip(layer.col, dout)
        |> Enum.map(fn {col, channel} ->
          channel
          |> Enum.map(fn data ->
            Enum.zip(col, data)
            |> Enum.map(fn {col2, row} ->
              Enum.zip(col2, row)
              |> Enum.map(fn {col3, val} ->
                col3
                |> Enum.map(&(&1 * val))
              end)
              |> Enum.reduce(
                List.duplicate(0.0, layer.filter_height * layer.filter_width),
                fn list, acc ->
                  Broca.NN.add(list, acc)
                end
              )
            end)
            |> Enum.reduce(
              List.duplicate(0.0, layer.filter_height * layer.filter_width),
              fn list, acc ->
                Broca.NN.add(list, acc)
              end
            )
          end)
        end)
        |> Enum.reduce(
          List.duplicate(0.0, layer.filter_height * layer.filter_width),
          fn list, acc ->
            Broca.NN.add(list, acc)
          end
        )

      dx =
        dout
        |> Enum.map(fn channel ->
          channel_map =
            Enum.zip(layer.params[:weight], channel)
            |> Enum.reduce(
              %{},
              fn {weights, row}, channel_map ->
                {_, row_map} =
                  row
                  |> Enum.reduce(
                    {0, channel_map},
                    fn col, {idx1, data_map} ->
                      {_, res_map2} =
                        col
                        |> Enum.reduce(
                          {0, data_map},
                          fn val, {idx2, col_map} ->
                            {_, res_map} =
                              weights
                              |> Enum.reduce(
                                {0, col_map},
                                fn weight, {w_idx, m} ->
                                  v = weight * val

                                  {w_idx + 1,
                                   Map.update(
                                     m,
                                     {idx1 + div(w_idx, layer.filter_width),
                                      idx2 + rem(w_idx, layer.filter_width)},
                                     v,
                                     fn map_val -> map_val + v end
                                   )}
                                end
                              )

                            {idx2 + 1, res_map}
                          end
                        )

                      {idx1 + 1, res_map2}
                    end
                  )

                row_map
              end
            )

          {max_h, max_w} = Enum.max(Map.keys(channel_map))

          [
            for h <- 0..max_h do
              for w <- 0..max_w do
                Map.get(channel_map, {h, w}, 0.0)
              end
            end
          ]
        end)

      {%Broca.Layers.Convolution{
         layer
         | grads: [weight: dw, bias: db],
           col: nil,
           activation: activation
       }, dx}
    end

    def update(layer, optimize_func) do
      updated_params =
        Keyword.keys(layer.params)
        |> Enum.reduce([], fn key, keyword ->
          Keyword.put_new(
            keyword,
            key,
            optimize_func.(Keyword.get(layer.params, key), Keyword.get(layer.grads, key))
          )
        end)

      %Broca.Layers.Convolution{layer | params: updated_params, grads: []}
    end

    def batch_update(layer1, layer2, optimize_func) do
      %Broca.Layers.Convolution{
        layer2
        | params: optimize_func.(layer1.params, layer2.grads),
          grads: []
      }
    end
  end
end
