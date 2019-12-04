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
            input_channel_size: 1,
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
        stride,
        padding,
        activation_type \\ nil,
        weight_init_std \\ 0.01
      ) do
    %Broca.Layers.Convolution{
      filter_height: height,
      filter_width: width,
      stride: stride,
      padding: padding,
      input_channel_size: input_channel_size,
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
      "Convolution: filter_size=#{conv.filter_height}x#{conv.filter_width}, input_channel_size = #{
        conv.input_channel_size
      }, stride=#{conv.stride}, padding=#{conv.padding}, " <>
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
      [[[[63, 72, 81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]],
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
      [[[[64, 73, 82], [109, 118, 127], [154, 163, 172]], [[128, 146, 164], [218, 236, 254], [308, 326, 344]]],
       [[[289, 298, 307], [334, 343, 352], [379, 388, 397]], [[578, 596, 614], [668, 686, 704], [758, 776, 794]]]]}

      iex> input = [[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], \
      [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]], \
      [[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]], \
      [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]]
      iex> conv = %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, input_channel_size: 2, params: [weight: [[0, 1, 2, 3, 4, 5, 6, 7, 8, \
      9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], bias: [1, 2, 3]]}
      iex> Layer.forward(conv, input)
      {%Broca.Layers.Convolution{filter_height: 3, filter_width: 3, input_channel_size: 2, params: [weight: [[0, 1, 2, 3, 4, 5, 6, 7, 8, \
      9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], bias: [1, 2, 3]], 
      col: [[[[0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26], [1, 2, 3, 5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23, 25, 26, 27]],
      [[4, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 22, 24, 25, 26, 28, 29, 30], [5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23, 25, 26, 27, 29, 30, 31]]],
      [[[32, 33, 34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58], [33, 34, 35, 37, 38, 39, 41, 42, 43, 49, 50, 51, 53, 54, 55, 57, 58, 59]],
      [[36, 37, 38, 40, 41, 42, 44, 45, 46, 52, 53, 54, 56, 57, 58, 60, 61, 62], [37, 38, 39, 41, 42, 43, 45, 46, 47, 53, 54, 55, 57, 58, 59, 61, 62, 63]]]]},
      [[[[2794, 2947], [3406, 3559]], [[7007, 7484], [8915, 9392]], [[11220, 12021], [14424, 15225]]],
       [[[7690, 7843], [8302, 8455]], [[22271, 22748], [24179, 24656]], [[36852, 37653], [40056, 40857]]]]
      }
    """
    def forward(layer, input) do
      s = System.os_time(:millisecond)

      col =
        Broca.NN.matrix_filtering(
          input,
          layer.filter_height,
          layer.filter_width,
          layer.stride,
          layer.padding
        )

      s1 = System.os_time(:millisecond)
      IO.puts("*** Convolution col: #{s1 - s}msec")

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

      s2 = System.os_time(:millisecond)
      IO.puts("** Convolution out: #{s2 - s1}msec")

      res =
        out
        |> Broca.NN.reshape([
          batch_size,
          out_height,
          out_width,
          div(total, batch_size * out_height * out_width)
        ])
        |> Broca.NN.transpose(0, 3, 1, 2)

      s3 = System.os_time(:millisecond)
      IO.puts("*** Convolution res: #{s3 - s2}msec")

      IO.puts("** Convolution forward: #{System.os_time(:millisecond) - s}msec")

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
          grads: [weight: [[159.20000000000002, 164.4, 169.6, 185.20000000000002, 190.4, 195.6, 211.20000000000002, 216.4, 221.6], 
          [231.20000000000002, 239.6, 248.0, 273.20000000000005, 281.6, 290.0, 315.20000000000005, 323.59999999999997, 332.0]], bias: [5.199999999999999, 8.4]]},
          [[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.1, 2.5, 2.5, 1.4], [0.0, 2.8, 6.199999999999999, 6.199999999999999, 3.4000000000000004],
          [0.0, 2.8, 6.199999999999999, 6.199999999999999, 3.4000000000000004], [0.0, 1.7, 3.6999999999999997, 3.6999999999999997, 2.0]]],
          [[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.5, 7.3, 7.3, 3.8], [0.0, 7.6, 15.8, 15.8, 8.2], [0.0, 7.6, 15.8, 15.8, 8.2], [0.0, 4.1, 8.5, 8.5, 4.4]]]]
        }

        iex> conv = %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, input_channel_size: 2, params: [weight: [[0, 1, 2, 3, 4, 5, 6, 7, 8, \
        9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
        [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], bias: [1, 2, 3]], \
        col: [[[[0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26], [1, 2, 3, 5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23, 25, 26, 27]], \
        [[4, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 22, 24, 25, 26, 28, 29, 30], [5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23, 25, 26, 27, 29, 30, 31]]], \
        [[[32, 33, 34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58], [33, 34, 35, 37, 38, 39, 41, 42, 43, 49, 50, 51, 53, 54, 55, 57, 58, 59]], \
        [[36, 37, 38, 40, 41, 42, 44, 45, 46, 52, 53, 54, 56, 57, 58, 60, 61, 62], [37, 38, 39, 41, 42, 43, 45, 46, 47, 53, 54, 55, 57, 58, 59, 61, 62, 63]]]]}
        iex> dout = [[[[0.0, 0.01], [0.02, 0.03]], [[0.04, 0.05], [0.06, 0.07]], [[0.08, 0.09], [0.1 , 0.11]]], [[[0.12, 0.13], [0.14, 0.15]], [[0.16, 0.17], \
        [0.18, 0.19]], [[0.2 , 0.21], [0.22, 0.23]]]]
        iex> Layer.backward(conv, dout)
        {%Broca.Layers.Convolution{filter_height: 3, filter_width: 3, input_channel_size: 2, params: [weight: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], bias: [1, 2, 3]],
        grads: [weight: [[18.959999999999997, 19.56, 20.159999999999997, 21.360000000000003, 21.959999999999997, 22.560000000000002, 23.759999999999998, 24.36,
         24.959999999999997, 28.56, 29.159999999999997, 29.759999999999998, 30.960000000000004, 31.56, 32.16, 33.36, 33.96, 34.56],
        [24.880000000000003, 25.799999999999997, 26.72, 28.56, 29.480000000000004, 30.400000000000002, 32.24, 33.16, 34.080000000000005, 39.599999999999994, 40.52,
         41.440000000000005, 43.279999999999994, 44.2, 45.120000000000005, 46.96000000000001, 47.88, 48.800000000000004],
        [30.799999999999997, 32.040000000000006, 33.279999999999994, 35.760000000000005, 37.0, 38.24, 40.72, 41.96, 43.2, 50.64, 51.879999999999995, 53.120000000000005,
         55.6 , 56.84, 58.08, 60.56, 61.8, 63.040000000000006]], bias: [0.6000000000000001 , 0.9199999999999999, 1.2400000000000002]]},
        [[[[3.5999999999999996, 7.859999999999999, 8.13, 4.4399999999999995], [8.64, 18.75, 19.41, 10.53], [9.540000000000001, 20.729999999999997, 21.39, 11.61],
        [5.76, 12.42, 12.810000000000002, 6.9]], [[4.68, 10.29, 10.559999999999999, 5.79], [11.34, 24.69, 25.349999999999998, 13.770000000000001],
        [12.24, 26.67, 27.330000000000002, 14.85], [7.380000000000001, 15.93, 16.32, 8.79]]],
        [[[10.08, 21.18, 22.17, 11.64], [22.68, 47.550000000000004, 49.65, 26.009999999999998], [25.740000000000002, 53.85000000000001, 55.95, 29.25],
        [14.4 , 30.060000000000002, 31.17, 16.26]], [[14.4 , 30.09, 31.080000000000002, 16.23], [31.86, 66.44999999999999, 68.55, 35.730000000000004],
        [34.92, 72.75, 74.85000000000001, 38.97], [19.259999999999998, 40.05, 41.16, 21.39]]]]
        }
    """
    def backward(layer, dout) do
      {activation, dout} =
        if not is_nil(layer.activation) do
          Layer.backward(layer.activation, dout)
        else
          {nil, dout}
        end

      s = System.os_time(:millisecond)

      db =
        dout
        |> Enum.reduce(
          List.duplicate(0.0, length(hd(dout))),
          fn batch, acc ->
            Enum.map(batch, &(Broca.NN.sum(&1) |> Enum.sum()))
            |> Broca.NN.add(acc)
          end
        )

      s1 = System.os_time(:millisecond)
      IO.puts("*** Convolution db: #{s1 - s}msec")

      dw =
        Enum.map(Enum.zip(layer.col, dout), fn {col, channel} ->
          Enum.map(channel, fn data ->
            Enum.zip(col, data)
            |> Enum.map(fn {col2, row} ->
              Enum.map(Enum.zip(col2, row), fn {col3, val} ->
                Enum.map(col3, &(&1 * val))
              end)
              |> Enum.reduce(
                List.duplicate(
                  0.0,
                  layer.input_channel_size * layer.filter_height * layer.filter_width
                ),
                fn list, acc ->
                  Broca.NN.add(list, acc)
                end
              )
            end)
            |> Enum.reduce(
              List.duplicate(
                0.0,
                layer.input_channel_size * layer.filter_height * layer.filter_width
              ),
              fn list, acc ->
                Broca.NN.add(list, acc)
              end
            )
          end)
        end)
        |> Enum.reduce(
          List.duplicate(
            0.0,
            layer.input_channel_size * layer.filter_height * layer.filter_width
          ),
          fn list, acc ->
            Broca.NN.add(list, acc)
          end
        )

      s2 = System.os_time(:millisecond)
      IO.puts("*** Convolution dw: #{s2 - s1}msec")

      dx =
        Enum.map(dout, fn channel ->
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
                            w_unit = div(length(weights), layer.input_channel_size)

                            {_, res_map} =
                              weights
                              |> Enum.reduce(
                                {0, col_map},
                                fn weight, {w_idx, m} ->
                                  v = weight * val

                                  {w_idx + 1,
                                   Map.update(
                                     m,
                                     {div(w_idx, w_unit),
                                      idx1 + div(rem(w_idx, w_unit), layer.filter_width),
                                      idx2 + rem(rem(w_idx, w_unit), layer.filter_width)},
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

          {max_c, max_h, max_w} = Enum.max(Map.keys(channel_map))

          for c <- 0..max_c do
            for h <- 0..max_h do
              for w <- 0..max_w do
                Map.get(channel_map, {c, h, w}, 0.0)
              end
            end
          end
        end)

      s3 = System.os_time(:millisecond)
      IO.puts("*** Convolution dx: #{s3 - s2}msec")

      IO.puts("** Convolution backward: #{System.os_time(:millisecond) - s}msec")

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
