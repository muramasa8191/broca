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

    def load_train_data(normalize \\ True, is_one_hot \\ True, flatten \\ True) do
      {parse_data(File.read!(Application.app_dir(:broca, "priv/" <> @train_data)), normalize, flatten),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @train_label)), is_one_hot)}
    end

    def load_test_data(normalize \\ True, is_one_hot \\ True, flatten \\ True) do
      {parse_data(File.read!(Application.app_dir(:broca, "priv/" <> @test_data)), normalize, flatten),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @test_label)), is_one_hot)}
    end

    defp parse_data(raw, normalize, flatten) do
      # IO.inspect(raw)
      <<_::unsigned-32, size::unsigned-32, rows::unsigned-32, cols::unsigned-32, bin::binary>> =
        raw

      # IO.puts("ID = #{id}")
      # IO.puts("# of image: #{size}")
      # IO.puts("rows x cols = #{rows} x #{cols}")
      image_size = rows * cols

      if flatten == True do
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

            if normalize == True do
              {[:erlang.binary_to_list(image) |> Enum.map(&(&1 / 255.0))] ++ acc, rest}
            else
              {[:erlang.binary_to_list(image)] ++ acc, rest}
            end
          end
        )

      data
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
                if normalize == True do
                  {[:erlang.binary_to_list(img) |> Enum.map(&(&1 / 255.0))] ++ list, raws}
                else
                  {[:erlang.binary_to_list(img)] ++ list, raws}
                end
              end)
            {[Enum.reverse(image)]++acc, rest}
          end
        )

      data
    end

    def parse_label(labels, is_one_hot \\ True) do
      <<_::unsigned-32, _::unsigned-32, bin::binary>> = labels
      # IO.puts("ID: #{id}")
      # IO.puts("size: #{size}")
      # IO.inspect(bin)
      if is_one_hot do
        :erlang.binary_to_list(bin) |> Enum.map(&Broca.NN.one_hot(&1, 9)) |> Enum.reverse()
      else
        :erlang.binary_to_list(bin) |> Enum.reverse()
      end
    end
  end
end
