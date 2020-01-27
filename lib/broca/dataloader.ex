defprotocol Broca.DataLoader do
  def get_batch_data(dataloader, batch_size)
  def size(dataloader)
end

defmodule Broca.DataLoader.CIFAR10.TrainNoCache do
  defstruct name: "cifar10_train"

  defimpl Broca.DataLoader, for: Broca.DataLoader.CIFAR10.TrainNoCache do
    def get_batch_data(_dataloader, batch_size) do
      {x_batch, t_batch} = Broca.Dataset.CIFAR10.load_batch_data(batch_size)

      Enum.zip(x_batch, t_batch)
      |> Enum.shuffle()
    end

    def size(_dataloader) do
      50000
    end
  end
end

defmodule Broca.DataLoader.CIFAR10.Validation do
  defstruct data: nil

  defimpl Broca.DataLoader, for: Broca.DataLoader.CIFAR10.Validation do
    def get_batch_data(dataloader, batch_size) do
      {x_train, t_train} =
        if is_nil(dataloader.data),
          do: Broca.Dataset.CIFAR10.load_test_data(),
          else: dataloader.data

      data =
        Enum.zip(x_train, t_train)
        |> Enum.take(batch_size)

      {data,
       %Broca.DataLoader.CIFAR10.Validation{
         dataloader
         | data: Enum.drop(dataloader.data, batch_size)
       }}
    end

    def size(_dataloader) do
      10000
    end
  end
end

defmodule Broca.DataLoader.MNIST.Train do
  defstruct data: nil

  def init() do
    %Broca.DataLoader.MNIST.Train{data: Broca.Dataset.MNIST.load_train_data()}
  end

  defimpl Broca.DataLoader, for: Broca.DataLoader.MNIST.Train do
    def get_batch_data(dataloader, batch_size) do
      {x_train, t_train} = dataloader.data

      Enum.zip(x_train, t_train)
      |> Enum.shuffle()
      |> Enum.take(batch_size)
    end

    def size(dataloader) do
      length(elem(dataloader.data, 1))
    end
  end
end

defmodule Broca.DataLoader.MNIST.Flatten.Train do
  defstruct data: nil

  def init() do
    %Broca.DataLoader.MNIST.Train{data: Broca.Dataset.MNIST.load_train_data(true, true, true)}
  end

  defimpl Broca.DataLoader, for: Broca.DataLoader.MNIST.Flatten.Train do
    def get_batch_data(dataloader, batch_size) do
      {x_train, t_train} = dataloader.data

      Enum.zip(x_train, t_train)
      |> Enum.shuffle()
      |> Enum.take(batch_size)
    end

    def size(dataloader) do
      length(elem(dataloader.data, 1))
    end
  end
end

defmodule Broca.DataLoader.MNIST.Validation do
  defstruct data: nil

  defimpl Broca.DataLoader, for: Broca.DataLoader.MNIST.Validation do
    def get_batch_data(dataloader, batch_size) do
      {x_test, t_test} =
        if is_nil(dataloader.data),
          do: Broca.Dataset.MNIST.load_test_data(),
          else: dataloader.data

      data =
        Enum.zip(x_test, t_test)
        |> Enum.take(batch_size)

      {data,
       %Broca.DataLoader.MNIST.Validation{
         data: {Enum.drop(x_test, batch_size), Enum.drop(t_test, batch_size)}
       }}
    end

    def size(_dataloader) do
      10000
    end
  end
end

defmodule Broca.DataLoaders do
  def create(type) do
    case type do
      :cifar10_train_no_cache -> %Broca.DataLoader.CIFAR10.TrainNoCache{}
      :cifar10_validation -> %Broca.DataLoader.CIFAR10.Validation{}
      :mnist_train -> Broca.DataLoader.MNIST.Train.init()
      :mnist_train_flatten -> Broca.DataLoader.MNIST.Flatten.Train.init()
      :mnist_validation -> %Broca.DataLoader.MNIST.Validation{}
    end
  end
end
