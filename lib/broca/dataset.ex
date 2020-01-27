defmodule Broca.Dataset.MNIST do
  @train_data "train-images-idx3-ubyte"
  @train_label "train-labels-idx1-ubyte"
  @test_data "t10k-images-idx3-ubyte"
  @test_label "t10k-labels-idx1-ubyte"

  alias Broca.Nif.NN

  def load_train_data(normalize \\ true, is_one_hot \\ true, flatten \\ false) do
    {parse_data(
       File.read!(Application.app_dir(:broca, "priv/" <> @train_data)),
       normalize,
       flatten
     ), parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @train_label)), is_one_hot)}
  end

  def load_test_data(normalize \\ true, is_one_hot \\ true, flatten \\ false) do
    {parse_data(
       File.read!(Application.app_dir(:broca, "priv/" <> @test_data)),
       normalize,
       flatten
     ), parse_label(File.read!(Application.app_dir(:broca, "priv/" <> @test_label)), is_one_hot)}
  end

  defp parse_data(raw, normalize, flatten) do
    <<_::unsigned-32, size::unsigned-32, rows::unsigned-32, cols::unsigned-32, bin::binary>> = raw

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
      Enum.map(:erlang.binary_to_list(img_binary), &(&1 / 255.0))
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

              {[normalize_data(img, normalize)] ++ list, raws}
            end)

          {[[Enum.reverse(image)]] ++ acc, rest}
        end
      )

    data
  end

  defp convert_one_hot(bin_label, is_one_hot) do
    if is_one_hot do
      :erlang.binary_to_list(bin_label) |> Enum.map(&NN.one_hot(&1, 9)) |> Enum.reverse()
    else
      :erlang.binary_to_list(bin_label) |> Enum.reverse()
    end
  end

  defp parse_label(labels, is_one_hot) do
    <<_::unsigned-32, _::unsigned-32, bin::binary>> = labels

    convert_one_hot(bin, is_one_hot)
  end
end

defmodule Broca.Dataset.CIFAR10 do
  alias Broca.{
    Dataset.CIFAR10,
    Naive.NN
  }

  require Logger

  @train_data "cifar10/data_batch_?.bin"
  @test_data "cifar10/test_batch.bin"

  def load_batch_data(batch_size) do
    n = 1..5 |> Enum.shuffle() |> Enum.take(1)
    batch = 0..9999 |> Enum.shuffle() |> Enum.take(batch_size) |> Enum.sort()

    task =
      Task.async(CIFAR10, :parse_batch, [
        File.read!(
          Application.app_dir(
            :broca,
            "priv/" <> String.replace(@train_data, "?", Integer.to_string(hd(n)))
          )
        ),
        batch
      ])

    Enum.unzip(Task.await(task, :infinity))
  end

  def load_train_data(normalize \\ true, is_one_hot \\ true) do
    Logger.debug("load CIFAR10 train data")

    5..1
    |> Enum.reduce(
      [],
      fn idx, list ->
        task =
          Task.async(CIFAR10, :parse, [
            File.read!(
              Application.app_dir(
                :broca,
                "priv/" <> String.replace(@train_data, "?", Integer.to_string(idx))
              )
            ),
            10000,
            normalize,
            is_one_hot
          ])

        Task.await(task, :infinity) ++ list
      end
    )
    |> Enum.unzip()
  end

  def load_test_data(normalize \\ true, is_one_hot \\ true) do
    Logger.debug("load CIFAR10 test data")

    task =
      Task.async(CIFAR10, :parse, [
        File.read!(Application.app_dir(:broca, "priv/" <> @test_data)),
        1000,
        normalize,
        is_one_hot
      ])

    Enum.unzip(Task.await(task, :infinity))
  end

  defp color_normalize(color, normalize) do
    if normalize do
      Enum.chunk_every(Enum.map(:erlang.binary_to_list(color), fn x -> x / 255.0 end), 32)
    else
      Enum.chunk_every(:erlang.binary_to_list(color), 32)
    end
  end

  def parse_batch(raw, batch_indices) do
    batch_indices
    |> Enum.reduce(
      {[], raw, -1},
      fn idx, {acc, bin, prev} ->
        diff = if prev != -1, do: (idx - prev - 1) * 3073, else: (idx - 0) * 3073

        <<_::binary-size(diff), class::unsigned-8, red::binary-size(1024),
          green::binary-size(1024), blue::binary-size(1024), rest::binary>> = bin

        red = color_normalize(red, true)
        green = color_normalize(green, true)
        blue = color_normalize(blue, true)

        {[
           {
             [red, green, blue],
             NN.one_hot(class, 9)
           }
         ] ++ acc, rest, idx}
      end
    )
    |> elem(0)
  end

  def parse(raw, size, normalize, is_one_hot) do
    1..size
    |> Enum.reduce(
      {[], raw},
      fn _, {acc, bin} ->
        <<class::unsigned-8, red::binary-size(1024), green::binary-size(1024),
          blue::binary-size(1024), rest::binary>> = bin

        red = color_normalize(red, normalize)
        green = color_normalize(green, normalize)
        blue = color_normalize(blue, normalize)

        {[
           {
             [red, green, blue],
             if(is_one_hot, do: NN.one_hot(class, 9), else: class)
           }
         ] ++ acc, rest}
      end
    )
    |> elem(0)
  end
end
