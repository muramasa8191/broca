defmodule Broca.Layers do
  alias Broca.{
    Random,
    Nif.NN,
    Optimizers,
    Layers
  }

  require Logger

  defp set_optimizer([layer_type, params], optimizer_type, learning_rate) do
    [
      layer_type,
      params,
      Optimizers.init(optimizer_type, params, learning_rate)
    ]
  end

  @doc """
  Set Optimizer for all applicable layers

  ## Examples
      iex> Broca.Layers.set_optimizers([[:relu]], :adam, 0.001)
      [[:relu]]
  """
  def set_optimizers(layers, optimizer_type, learning_rate) do
    Enum.map(
      layers,
      fn layer ->
        case layer do
          [layer_type, params, fix_params] ->
            set_optimizer([layer_type, params], optimizer_type, learning_rate) ++ [fix_params]

          [:max_pooling2d, _] ->
            layer

          [:dropout, _] ->
            layer

          [_layer_type, _params] ->
            set_optimizer(layer, optimizer_type, learning_rate)

          [layer_type] ->
            [layer_type]
        end
      end
    )
  end

  @doc """
  Generate Affine Layer list

  ## Example
      iex> affine = Broca.Layers.affine(10, 1)
      iex> hd affine
      :affine
  """
  def affine(input_size, output_size) do
    [
      :affine,
      [
        weight:
          Random.randn(input_size, output_size)
          |> NN.mult(:math.sqrt(2.0 / input_size)),
        bias: List.duplicate(0.0, output_size)
      ]
    ]
  end

  @doc """
  Generate ReLU Activation list

  ## Example
      iex> Broca.Layers.relu
      [:relu]
  """
  def relu() do
    [:relu]
  end

  @doc """
  Generate Softmax Activation list

  ## Example
      iex> Broca.Layers.softmax()
      [:softmax]
  """
  def softmax() do
    [:softmax]
  end

  @doc """
  Generate Convolution2D

  ## Example
      iex> conv = Broca.Layers.convolution2d(3, 3, 3, 32, 1, 1)
      iex> hd conv
      :conv2d
  """
  def convolution2d(
        height,
        width,
        input_channel_size,
        filter_size,
        stride,
        padding,
        weight_init_std \\ 0.01
      ) do
    [
      :conv2d,
      [
        weight:
          Broca.Random.randn(filter_size, input_channel_size * width * height)
          |> NN.mult(weight_init_std),
        bias: List.duplicate(0.0, filter_size)
      ],
      [
        filter_height: height,
        filter_width: width,
        stride: stride,
        padding: padding,
        input_channel_size: input_channel_size
      ]
    ]
  end

  @doc """
  Generate Max Pooling2D

  """
  def max_pooling2d(height, width, stride, padding) do
    [
      :max_pooling2d,
      [
        pool_height: height,
        pool_width: width,
        stride: stride,
        padding: padding
      ]
    ]
  end

  @doc """
  Generate Dropout

  ## Example
      iex> Broca.Layers.dropout(0.25)
      [:dropout, [ratio: 0.25]]
  """
  def dropout(ratio) do
    [:dropout, [ratio: ratio]]
  end

  defp create_dropout_mask([n, c, h, w], ratio) do
    for _ <- 1..n do
      for _ <- 1..c do
        for _ <- 1..h do
          for _ <- 1..w do
            if :rand.uniform() < ratio, do: 0.0, else: 1.0
          end
        end
      end
    end
  end

  defp create_dropout_mask([h, w], ratio) do
    for _ <- 1..h do
      for _ <- 1..w do
        if :rand.uniform() < ratio, do: 0.0, else: 1.0
      end
    end
  end

  defp affine_preprocessing(x, dimention) when dimention > 2 do
    shape = NN.shape(x)
    NN.reshape(x, [hd(shape), tl(shape) |> Enum.reduce(1, fn x, acc -> acc * x end)])
  end

  defp affine_preprocessing(x, _) do
    x
  end

  defp get_conv_forward_shape(x, filter_height, filter_width, padding, stride, filter_size) do
    [n, _, h, w] = NN.shape(x)
    out_height = 1 + div(h + 2 * padding - filter_height, stride)
    out_width = 1 + div(w + 2 * padding - filter_width, stride)
    [n, out_height, out_width, filter_size]
  end

  @doc """
  Forward Propagation for Affine Layer

  """
  def forward(x, [:affine, params]) do
    original_shape = NN.shape(x)
    x = affine_preprocessing(x, NN.shape(x))
    out = NN.dot(x, params[:weight]) |> NN.add(params[:bias])

    {out, [x: x, original_shape: original_shape]}
  end

  @doc """
  Forward Propagation for ReLU

  """
  def forward(x, :relu) do
    mask = NN.for_each(x, fn val -> if val < 0.0, do: 0.0, else: 1.0 end)
    out = NN.mult(x, mask)

    {out, [mask: mask]}
  end

  @doc """
  Forward Propagation for Softmax

  """
  def forward(x, :softmax) do
    out = NN.softmax(x)

    {out, [out: out]}
  end

  @doc """
  Forward Propagation for Convolution2D

  """
  def forward(x, [:conv2d, params, fix_params]) do
    task =
      Task.async(NN, :matrix_filtering, [
        x,
        fix_params[:filter_height],
        fix_params[:filter_width],
        fix_params[:stride],
        fix_params[:padding]
      ])

    col = Task.await(task, :infinity)
    size = NN.data_size(col)
    target_size = List.last(NN.shape(col))
    col = NN.reshape(col, [div(size, target_size), target_size])

    res =
      NN.dot_nt(col, params[:weight])
      |> NN.add(params[:bias])
      |> NN.reshape(
        get_conv_forward_shape(
          x,
          fix_params[:filter_height],
          fix_params[:filter_width],
          fix_params[:padding],
          fix_params[:stride],
          length(params[:weight])
        )
      )
      |> Broca.Nif.NN.transpose(0, 3, 1, 2)

    {res, [col: col]}
  end

  @doc """
  Forward Propagation for MaxPooling2D

  """
  def forward(x, [:max_pooling2d, fix_params]) do
    res =
      NN.matrix_filtering(
        x,
        fix_params[:pool_height],
        fix_params[:pool_width],
        fix_params[:stride],
        fix_params[:padding],
        fn list -> Enum.max(list) end,
        :normal
      )

    mask =
      NN.matrix_filtering(
        x,
        fix_params[:pool_height],
        fix_params[:pool_width],
        fix_params[:stride],
        fix_params[:padding],
        fn list -> NN.argmax(list) end,
        :normal
      )

    [_, _, height, width] = NN.shape(x)

    {res, [mask: mask, original_height: height, original_width: width]}
  end

  @doc """
  Forward Propagation for Dropout

  """
  def forward(x, [:dropout, fix_params]) do
    mask = create_dropout_mask(NN.shape(x), fix_params[:ratio])

    {NN.mult(mask, x), [mask: mask]}
  end

  def get_conv_dx(dout, params, fix_params) do
    Enum.map(dout, fn channel ->
      channel_map =
        Enum.zip(params[:weight], channel)
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
                        w_unit = div(length(weights), fix_params[:input_channel_size])

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
                                  idx1 + div(rem(w_idx, w_unit), fix_params[:filter_width]) -
                                    fix_params[:padding],
                                  idx2 + rem(rem(w_idx, w_unit), fix_params[:filter_width]) -
                                    fix_params[:padding]},
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
        for h <- 0..(max_h - fix_params[:padding]) do
          for w <- 0..(max_w - fix_params[:padding]) do
            Map.get(channel_map, {c, h, w}, 0.0)
          end
        end
      end
    end)
  end

  def maxpooling_backward(dout, fix_params, config) do
    Enum.zip(config[:mask], dout)
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
                      (div(mask, fix_params[:pool_width]) + idx1) * config[:original_width] +
                        idx2 +
                        rem(mask, fix_params[:pool_width])

                    {idx2 + fix_params[:stride], Map.update(map, idx, val, fn v -> v + val end)}
                  end
                )

              {idx1 + fix_params[:stride], result_map}
            end
          )

        for y <- 0..(config[:original_height] - 1) do
          for x <- 0..(config[:original_width] - 1) do
            idx = y * config[:original_width] + x
            Map.get(result, idx, 0.0)
          end
        end
      end)
    end)
  end

  @doc """
  Backward Propagation 

  ### Examples
      iex> dout = [[[[0.0, 0.01], [0.02, 0.03]], [[0.04, 0.05], [0.06, 0.07]], [[0.08, 0.09], [0.1 , 0.11]]], [[[0.12, 0.13], [0.14, 0.15]], [[0.16, 0.17], \
      [0.18, 0.19]], [[0.2 , 0.21], [0.22, 0.23]]]]
      iex> conv = [:conv2d, [weight: [[0, 1, 2, 3, 4, 5, 6, 7, 8, \
      9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], \
      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], bias: [1, 2, 3]], \
      [filter_height: 3, filter_width: 3, input_channel_size: 2, stride: 1, padding: 0], \
      [col: [[[[0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26], [1, 2, 3, 5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23, 25, 26, 27]], \
      [[4, 5, 6, 8, 9, 10, 12, 13, 14, 20, 21, 22, 24, 25, 26, 28, 29, 30], [5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23, 25, 26, 27, 29, 30, 31]]], \
      [[[32, 33, 34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58], [33, 34, 35, 37, 38, 39, 41, 42, 43, 49, 50, 51, 53, 54, 55, 57, 58, 59]], \
      [[36, 37, 38, 40, 41, 42, 44, 45, 46, 52, 53, 54, 56, 57, 58, 60, 61, 62], [37, 38, 39, 41, 42, 43, 45, 46, 47, 53, 54, 55, 57, 58, 59, 61, 62, 63]]]]]]
      iex> Broca.Layers.backward(dout, conv)
      {[[[[3.5999999999999996, 7.859999999999999, 8.13, 4.4399999999999995], [8.64, 18.75, 19.41, 10.53], [9.540000000000001, 20.729999999999997, 21.39, 11.61],
      [5.76, 12.42, 12.810000000000002, 6.9]], [[4.68, 10.29, 10.559999999999999, 5.79], [11.34, 24.69, 25.349999999999998, 13.770000000000001],
      [12.24, 26.67, 27.330000000000002, 14.85], [7.380000000000001, 15.93, 16.32, 8.79]]],
      [[[10.08, 21.18, 22.17, 11.64], [22.68, 47.550000000000004, 49.65, 26.009999999999998], [25.740000000000002, 53.85000000000001, 55.95, 29.25],
      [14.4 , 30.060000000000002, 31.17, 16.26]], [[14.4 , 30.09, 31.080000000000002, 16.23], [31.86, 66.44999999999999, 68.55, 35.730000000000004],
      [34.92, 72.75, 74.85000000000001, 38.97], [19.259999999999998, 40.05, 41.16, 21.39]]]],
      [weight: [[18.959999999999997, 19.56, 20.159999999999997, 21.360000000000003, 21.959999999999997, 22.560000000000002, 23.759999999999998, 24.36,
        24.959999999999997, 28.56, 29.159999999999997, 29.759999999999998, 30.960000000000004, 31.56, 32.16, 33.36, 33.96, 34.56],
      [24.880000000000003, 25.799999999999997, 26.72, 28.56, 29.480000000000004, 30.400000000000002, 32.24, 33.16, 34.080000000000005, 39.599999999999994, 40.52,
        41.440000000000005, 43.279999999999994, 44.2, 45.120000000000005, 46.96000000000001, 47.88, 48.800000000000004],
      [30.799999999999997, 32.040000000000006, 33.279999999999994, 35.760000000000005, 37.0, 38.24, 40.72, 41.96, 43.2, 50.64, 51.879999999999995, 53.120000000000005,
        55.6 , 56.84, 58.08, 60.56, 61.8, 63.040000000000006]], bias: [0.6000000000000001 , 0.9199999999999999, 1.2400000000000002]]}

      iex> dout = [[[[0.31, 0.51, 0.59, 0.61],[0.52, 0.55, 0.03, 0.34], [0.06, 0.05, 0.16, 0.11], [0.14, 0.28, 0.44, 0.57]], \
      [[0.17, 0.1, 0.63, 0.48], [0.45, 0.22, 0.33, 0.6], [0.5, 0.56, 0.19, 0.54], [0.62, 0.04, 0.32, 0.38]]], \
      [[[0.12, 0.21, 0.49, 0.37], [0.53, 0.43, 0.13, 0.01], [0.15, 0.2, 0.35, 0.46], [0.26, 0.24, 0.23, 0.41]], \
      [[0.25, 0.42, 0.36, 0.18], [0.39, 0.27, 0.47, 0.0], [0.08, 0.07, 0.3 , 0.09], [0.02, 0.58, 0.29, 0.4 ]]]]
      iex> conv = [:conv2d, [weight: [[1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2]], bias: [0, 0]], [filter_height: 2, filter_width: 2, padding: 1, input_channel_size: 2], \
      [col: [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09], [0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.09, 0.1], \
      [0.0, 0.0, 0.01, 0.02, 0.0, 0.0, 0.1, 0.11], [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.11, 0.0]], \
      [[0.0, 0.0, 0.0, 0.03, 0.0, 0.09, 0.0, 0.12],[0.0, 0.01, 0.03, 0.04, 0.09, 0.1, 0.12, 0.13], [0.01, 0.02, 0.04, 0.05, 0.1, 0.11, 0.13, 0.14], [0.02, 0.0, 0.05, 0.0, 0.11, 0.0, 0.14, 0.0]], \
      [[0.0, 0.03, 0.0, 0.06, 0.0, 0.12, 0.0, 0.15], [0.03, 0.04, 0.06, 0.07, 0.12, 0.13, 0.15, 0.16], [0.04, 0.05, 0.07, 0.08, 0.13, 0.14, 0.16, 0.17], [0.05, 0.0, 0.08, 0.0, 0.14, 0.0, 0.17, 0.0]], \
      [[0.0, 0.06, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0], [0.06, 0.07, 0.0, 0.0, 0.15, 0.16, 0.0, 0.0], [0.07, 0.08, 0.0, 0.0, 0.16, 0.17, 0.0, 0.0], [0.08, 0.0, 0.0, 0.0, 0.17, 0.0, 0.0, 0.0]]], \
      [[[0.0, 0.0, 0.0, 0.18, 0.0, 0.0, 0.0, 0.27], [0.0, 0.0, 0.18, 0.19, 0.0, 0.0, 0.27, 0.28], [0.0, 0.0, 0.19, 0.2, 0.0, 0.0, 0.28, 0.29], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.29, 0.0]], \
      [[0.0, 0.18, 0.0, 0.21, 0.0, 0.27, 0.0, 0.3], [0.18, 0.19, 0.21, 0.22, 0.27, 0.28, 0.3, 0.31], [0.19, 0.2, 0.22, 0.23, 0.28, 0.29, 0.31, 0.32], [0.2, 0.0, 0.23, 0.0, 0.29, 0.0, 0.32, 0.0]], \
      [[0.0, 0.21, 0.0, 0.24, 0.0, 0.3, 0.0, 0.33], [0.21, 0.22, 0.24, 0.25, 0.3, 0.31, 0.33, 0.34], [0.22, 0.23, 0.25, 0.26, 0.31, 0.32, 0.34, 0.35], [0.23, 0.0, 0.26, 0.0, 0.32, 0.0, 0.35, 0.0]], \
      [[0.0, 0.24, 0.0, 0.0, 0.0, 0.33, 0.0, 0.0],[0.24, 0.25, 0.0, 0.0, 0.33, 0.34, 0.0, 0.0],[0.25, 0.26, 0.0, 0.0, 0.34, 0.35, 0.0, 0.0], [0.26, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0]]]]]]
      iex> Broca.Layers.backward(dout, conv)
      {
        [[[[3.77, 4.24, 5.65], [4.640000000000001, 3.39, 3.96], [3.9700000000000006, 3.1500000000000004, 4.14]], [[3.77, 4.24, 5.65],
          [4.640000000000001, 3.39, 3.96], [3.9700000000000006, 3.1500000000000004, 4.14]]],
        [[[3.95, 4.299999999999999, 3.02], [2.93, 3.33, 2.67], [2.3499999999999996, 3.5 , 3.6100000000000003]],
          [[3.95, 4.299999999999999, 3.02], [2.93, 3.33, 2.67], [2.3499999999999996, 3.5, 3.6100000000000003]]]],
        [weight: [[0.6643, 0.6224000000000001, 0.6569999999999999, 0.6482, 1.1134, 1.0499, 1.161 , 1.1333000000000002],
          [0.6769, 0.6631999999999999, 0.6111, 0.6977, 1.1854, 1.1762, 1.134, 1.2161]], bias: [9.86, 10.299999999999999]]
      }
  """
  def backward(dout, [:affine, params, config]) do
    grads = [
      weight: NN.dot_tn(config[:x], dout),
      bias: NN.sum(dout, :col)
    ]

    dx = NN.dot_nt(dout, params[:weight])

    {NN.reshape(dx, config[:original_shape]), grads}
  end

  @doc """
  Backward Propagation for ReLU

  ### Example
      iex> dout = [[100.0, 20.0, 30.0, 24.0]]
      iex> Broca.Layers.backward(dout, [:relu, [mask: [[0.0, 1.0, 1.0, 0.0]]]])
      {[[0.0, 20.0, 30.0, 0.0]], [:relu, nil, nil]}

      iex> dout = [[100.0, 20.0, 30.0, 24.0], [100.0, 20.0, 30.0, 24.0]]
      iex> Broca.Layers.backward(dout, [:relu, [mask: [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]])
      [[0.0, 20.0, 30.0, 0.0], [0.0, 20.0, 30.0, 0.0]]
  """
  def backward(dout, [:relu, [mask: mask]]) do
    NN.mult(dout, mask)
  end

  @doc """
  Backward Propagation for Softmax

  ### Example
      iex> dout = [[0, 1, 0],[0, 0, 1]]
      iex> Broca.Layers.backward(dout, [:softmax, [out: [[0.1, 0.5, 0.4], [0.2, 0.2, 0.6]]]])
      [[0.05, -0.25, 0.2], [0.1, 0.1, -0.2]]
  """
  def backward(dout, [:softmax, [out: out]]) do
    NN.subtract(out, dout) |> NN.division(length(dout))
  end

  @doc """
  Backward Propagation for Convolution2D

  """
  def backward(dout, [:conv2d, params, fix_params, config]) do
    s = System.os_time(:millisecond)

    dout_t =
      Broca.Nif.NN.transpose(dout, 0, 2, 3, 1)
      |> NN.reshape([div(NN.data_size(dout), length(params[:weight])), length(params[:weight])])

    s1 = System.os_time(:millisecond)
    Logger.debug("dout_t : #{s1 - s}msec")
    db = NN.sum(dout_t, :col)
    s2 = System.os_time(:millisecond)

    dw =
      NN.dot_tn(config[:col], dout_t)
      |> Broca.Nif.NN.transpose()

    s3 = System.os_time(:millisecond)
    Logger.debug("dw : #{s3 - s2}msec")

    task = Task.async(Layers, :get_conv_dx, [dout, params, fix_params])
    dx = Task.await(task, :infinity)
    s4 = System.os_time(:millisecond)
    Logger.debug("dx : #{s4 - s3}msec")
    {dx, [weight: dw, bias: db]}
  end

  @doc """
  Backward Propagation for MaxPooling2D

  ### Examples
      iex> dout = [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], [[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]]]]
      iex> pool = [:max_pooling2d, [pool_height: 2, pool_width: 2, stride: 1, padding: 0], \
      [original_height: 3, original_width: 3, mask: [[[[3, 3], [3, 3]], [[3, 3], [3, 3]]], [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]]]]
      iex> Broca.Layers.backward(dout, pool)
      [[[[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.3, 0.4]],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.6], [0.0, 0.7, 0.8]]],
        [[[0.0, 0.0, 0.0], [0.0, 0.9, 1.0], [0.0, 1.1, 1.2]],
        [[0.0, 0.0, 0.0], [0.0, 1.3, 1.4], [0.0, 1.5, 1.6]]]]

      iex> input = [[[[0.86157712,  -0.47107838, -1.48115791, 1.936075], [1.23430433,  -0.5916912, -0.33619328, -1.47721128], \
      [0.75957884,  0.39067426,  -0.3738926,  1.08552397], [-0.35477994, 1.37320599,  -0.50188206, -0.31016749]], \
      [[-1.23584931, 1.23890113,  1.32787184,  -1.39848664], [0.22482817,  -1.37082643, 0.25703018,  0.78216186], \
      [0.82867592,  0.66301564,  -0.65142784, -0.41643955], [0.07892943,  0.85376998,  -0.34785806, -0.61405054]]]]
      iex> pool = [:max_pooling2d, [pool_height: 2, pool_width: 2, stride: 2, padding: 0]]
      iex> {_, config} = Broca.Layers.forward(input, pool)
      iex> dout = [[[[0.07037291, -0.54616878], [-0.91264118, -1.62252685]], [[0.70192154, 0.13682912], [-0.61967839, 0.25081034]]]]
      iex> [type, fix_params] = pool
      iex> Broca.Layers.backward(dout, [type, fix_params, config])
      [[[[0.0, 0.0, 0.0, -0.54616878], [0.07037291, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.62252685], [0.0, -0.91264118, 0.0, 0.0]],
      [[0.0, 0.70192154, 0.13682912, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, -0.61967839, 0.25081034, 0.0]]]]

  """
  def backward(dout, [:max_pooling2d, fix_params, config]) do
    task = Task.async(Layers, :maxpooling_backward, [dout, fix_params, config])
    Task.await(task, :infinity)
  end

  @doc """
  Backward Propagation for Dropout

  ### Examples
      iex> dout = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]
      iex> dropout = [:dropout, [ratio: 0.25], [mask: [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]]]]
      iex> Broca.Layers.backward(dout, dropout)
      [[[[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]]]]
  """
  def backward(dout, [:dropout, _, config]) do
    NN.mult(config[:mask], dout)
  end
end
