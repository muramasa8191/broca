defmodule Broca do
  @moduledoc """
  Documentation for Broca.
  """

  def ch05(iterate, batch_size) do
    {x_train, t_train} = Broca.Dataset.MNIST.load_train()
    {model, loss_layer} = Broca.Models.TwoLayerNet.new(784, 50, 10)

    zip_data = Enum.zip(x_train, t_train)

    for i <- 1..iterate do
      {x_batch, t_batch} =
        zip_data
        |> Enum.shuffle()
        |> Enum.take(batch_size)
        |> Enum.unzip()

      {model2, out} = Broca.Models.TwoLayerNet.predict(model, x_batch)
      acc = Broca.Models.TwoLayerNet.accuracy(out, t_batch)
      IO.puts("Accuracy: #{acc}")

      {model3, _} = Broca.Models.TwoLayerNet.gradient(model2, loss_layer, x_batch, t_batch)
      model = Broca.Models.TwoLayerNet.update(model3)
    end
  end
end
