defmodule Broca do
  @moduledoc """
  Documentation for Broca.
  """
  def ch05(epochs, batch_size, learning_rate \\ 0.1) do
    {x_train, t_train} = Broca.Dataset.MNIST.load_train_data()
    {x_test, t_test} = Broca.Dataset.MNIST.load_test_data()
    {model, loss_layer} = Broca.Models.TwoLayerNet.new(784, 50, 10)
    data_size = length(x_train)
    IO.puts("Train on #{data_size} samples, Validation on #{length(x_test)} samples.")
    zip_data = Enum.zip(x_train, t_train)

    iterate = round(data_size / batch_size) |> max(1)
    optimizer = %Broca.Optimizers.SGD{}

    1..epochs
    |> Enum.reduce({model, optimizer}, fn epoch, {e_model, e_optimizer} ->
      IO.puts("Epoch #{epoch}/#{epochs}")

      {epoch_model, epoch_optimizer} =
        1..iterate
        |> Enum.reduce({e_model, e_optimizer}, fn i, {loop_model, loop_optimizer} ->
          if i == 1, do: IO.puts("0/#{iterate} [                    ]")

          {x_batch, t_batch} =
            zip_data
            |> Enum.shuffle()
            |> Enum.take(batch_size)
            |> Enum.unzip()

          grad_model = Broca.Models.TwoLayerNet.gradient(loop_model, loss_layer, x_batch, t_batch)

          {updated_model, updated_optimizer} =
            Broca.Models.TwoLayerNet.update(grad_model, loop_optimizer, learning_rate)

          loss = Broca.Models.TwoLayerNet.loss(updated_model, loss_layer, x_batch, t_batch)

          progress = round(i / iterate * 10)

          if i != iterate do
            IO.puts(
              "\e[1A#{i}/#{iterate} [#{
                if progress != 0, do: List.to_string(for _ <- 1..(progress * 2), do: "=")
              }#{List.to_string(for _ <- 1..((10 - progress) * 2), do: " ")}] - loss: #{
                Float.floor(loss, 5)
              }          "
            )
          else
            acc = Broca.Models.TwoLayerNet.accuracy(updated_model, x_batch, t_batch)
            test_acc = Broca.Models.TwoLayerNet.accuracy(updated_model, x_test, t_test)
            test_loss = Broca.Models.TwoLayerNet.loss(updated_model, loss_layer, x_test, t_test)

            IO.puts(
              "\e[1A#{i}/#{iterate} [#{
                if progress != 0, do: List.to_string(for _ <- 1..(progress * 2), do: "=")
              }] - loss: #{Float.floor(loss, 5)} - acc: #{Float.floor(acc, 5)} - test_loss: #{
                Float.floor(test_loss, 5)
              } - test_acc: #{Float.floor(test_acc, 5)}"
            )
          end

          {updated_model, updated_optimizer}
        end)

      {epoch_model, epoch_optimizer}
    end)
  end
end
