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

    def load_train_data(normalize \\ True, is_one_hot \\ True) do
      {parse_data(File.read!(Application.app_dir(:broca, "priv/" <> @train_data)), normalize),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @train_label)), is_one_hot)}
    end

    def load_test_data(normalize \\ True, is_one_hot \\ True) do
      {parse_data(File.read!(Application.app_dir(:broca, "priv/" <> @test_data)), normalize),
       parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @test_label)), is_one_hot)}
    end

    def parse_data(raw, normalize \\ True) do
      # IO.inspect(raw)
      <<_::unsigned-32,
        size::unsigned-32,
        rows::unsigned-32,
        cols::unsigned-32,
        bin::binary>> = raw
      # IO.puts("ID = #{id}")
      # IO.puts("# of image: #{size}")
      # IO.puts("rows x cols = #{rows} x #{cols}")
      image_size = rows * cols
      {data, _} =
      1..size
      |> Enum.reduce({[], bin}, 
                     fn _, {acc, rest} ->
                        <<image::binary-size(image_size), rest::binary>> = rest
                        if normalize do
                          {[:erlang.binary_to_list(image) |>Enum.map(&(&1/255.0))]++acc, rest}
                        else
                          {[:erlang.binary_to_list(image)]++acc, rest}
                        end 
                      end)

      data
    end

    def parse_label(labels, is_one_hot \\ True) do
      <<_::unsigned-32,
        _::unsigned-32,
        bin::binary>> = labels
      # IO.puts("ID: #{id}")
      # IO.puts("size: #{size}")
      # IO.inspect(bin)
      if is_one_hot do
        :erlang.binary_to_list(bin) |>Enum.map(&(Broca.NN.one_hot(&1, 9)))
      else
        :erlang.binary_to_list(bin)
      end
    end
  end

end