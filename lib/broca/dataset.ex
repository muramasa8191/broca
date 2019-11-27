defmodule Broca.Dataset do
  @moduledoc """
  Module to manipulate datasets.
  The sub modules are
  - MNIST

  @note no doctest due to too long execution time
  """
  defmodule MNIST do
    @train_data "train-images-idx3-ubyte"
    @train_label "train-labels-idx1-ubyte"
    @test_data "t10k-images-idx3-ubyte"
    @test_label "t10k-labels-idx1-ubyte"

    def load_train_data(normalize \\ true, is_one_hot \\ true, flatten \\ false) do
      {parse_data(
         File.read!(Application.app_dir(:broca, "priv/" <> @train_data)),
         normalize,
         flatten
       ),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @train_label)), is_one_hot)}
    end

    def load_test_data(normalize \\ true, is_one_hot \\ true, flatten \\ false) do
      {parse_data(
         File.read!(Application.app_dir(:broca, "priv/" <> @test_data)),
         normalize,
         flatten
       ),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @test_label)), is_one_hot)}
    end

    defp parse_data(raw, normalize, flatten) do
      <<_::unsigned-32, size::unsigned-32, rows::unsigned-32, cols::unsigned-32, bin::binary>> =
        raw

      image_size = rows * cols

      if flatten do
        retrieve_flat_data(bin, size, image_size, normalize)
      else
        retrieve_data(bin, size, rows, cols, normalize)
      end
    end

    defp retrieve_flat_data(bin, data_size, image_size, normalize) do
      {data, _} =
        1..data_size
        |> Enum.reduce(
          {[], bin},
          fn _, {acc, rest} ->
            <<image::binary-size(image_size), rest::binary>> = rest

            if normalize do
              {[:erlang.binary_to_list(image) |> Enum.map(&(&1 / 255.0))] ++ acc, rest}
            else
              {[:erlang.binary_to_list(image)] ++ acc, rest}
            end
          end
        )

      data
    end

    defp normalize_data(img_binary, normalize) do
      if normalize do
        Enum.map(:erlang.binary_to_list(img_binary),&(&1 / 255.0))
      else
        :erlang.binary_to_list(img_binary)
      end
    end

    defp retrieve_data(bin, data_size, rows, cols, normalize) do
      {data, _} =
        1..data_size
        |> Enum.reduce(
          {[], bin},
          fn _, {acc, rest} ->
            {image, rest} =
              1..rows
              |> Enum.reduce({[], rest}, fn _, {list, raws} ->
                <<img::binary-size(cols), raws::binary>> = raws

                {[normalize_data(img, normalize)]++list, raws}
              end)

            {[[Enum.reverse(image)]] ++ acc, rest}
          end
        )

      data
    end

    defp convert_one_hot(bin_label, is_one_hot) do
      if is_one_hot do
        :erlang.binary_to_list(bin_label) |> Enum.map(&Broca.NN.one_hot(&1, 9)) |> Enum.reverse()
      else
        :erlang.binary_to_list(bin_label) |> Enum.reverse()
      end
    end

    defp parse_label(labels, is_one_hot) do
      <<_::unsigned-32, _::unsigned-32, bin::binary>> = labels

      convert_one_hot(bin, is_one_hot)
    end
  end
end

defmodule Broca.Dataset.CIFAR10 do
  @train_data "cifar10/data_batch_?.bin"
  @test_data "cifar10/test_batch.bin"

  def load_train_data(normalize \\ true, is_one_hot \\ true) do
    5..1
    |> Enum.reduce([],
      fn idx, list ->
        [parse(
          File.read!(
            Application.app_dir(:broca, "priv/" <> String.replace(@train_data, "?", Integer.to_string(idx)))
          ),
           10000, normalize, is_one_hot)] ++ list
      end)
    |> Enum.unzip
  end

  def load_test_data(normalize \\ true, is_one_hot \\ true) do
    parse(
      File.read!(Application.app_dir(:broca, "priv/" <> @test_data)),
      1000,
      normalize,
      is_one_hot
    )
    |> Enum.unzip
  end

  defp color_normalize(color, normalize) do
    if normalize do
      Enum.chunk_every(Enum.map(:erlang.binary_to_list(color), fn x -> x / 255.0 end), 32)
    else
      Enum.chunk_every(:erlang.binary_to_list(color), 32)
    end
  end

  defp parse(raw, size, normalize, is_one_hot) do
    1..size
    |> Enum.reduce({[], raw},
      fn idx, {acc, bin} ->
        <<class::unsigned-8, 
          red::binary-size(1024),
          green::binary-size(1024),
          blue::binary-size(1024),
          rest::binary>> = bin

        red = color_normalize(red, normalize)
        green = color_normalize(green, normalize)
        blue = color_normalize(blue, normalize)
        
        {[{
          [red, green, blue,],
          (if is_one_hot, do: Broca.NN.one_hot(class, 9), else: class)
          }] ++ acc,
          rest
        }
        # rgb =
        #   Enum.zip(:erlang.binary_to_list(red), :erlang.binary_to_list(green))
        #   |> Enum.zip(:erlang.binary_to_list(blue))
        
        # {image, _} =
        #   1..32
        #   |> Enum.reduce({[], rgb},
        #     fn _, {row, rgb_row} ->
        #       {col_res, rest_rgb} =
        #         1..32
        #         |> Enum.reduce({[], rgb_row},
        #           fn _, {col, rgb_col} ->
        #             {{r, g}, b} = hd rgb_col
        #             if normalize == True do
        #               {[[r / 255, g / 255, b / 255]] ++ col, (tl rgb_col)}
        #             else
        #               {[[r, g, b]] ++ col, (tl rgb_col)}
        #             end
        #           end)
        #       {[Enum.reverse(col_res)] ++ row, rest_rgb}
        #   end)
        # {
        #   [{Enum.reverse(image), (if is_one_hot == True, do: Broca.NN.one_hot(class, 9), else: class)}] ++ acc,
        #   rest
        # }
      end)
    |> elem(0)
  end
end