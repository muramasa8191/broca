defmodule Broca.Trainer do

  def train(model, loss_layer, optimizer, {x_train, t_train}, epochs, batch_size, learning_rate, parallelize \\ False, test_data \\ nil) do
    data_size = length(x_train)
    if not is_nil(test_data) do
      IO.puts("Train on #{data_size} samples, Validation on #{length(elem(test_data, 0))} samples.")
    else 
      IO.puts("Train on #{data_size} samples.")
    end
    iterate = round(data_size / batch_size) |> max(1)

    1..epochs
    |> Enum.reduce({model, optimizer}, fn epoch, {e_model, e_optimizer} ->
      IO.puts("Epoch #{epoch}/#{epochs}")

      {epoch_model, epoch_optimizer} =
        1..iterate
        |> Enum.reduce({e_model, e_optimizer}, fn i, {loop_model, loop_optimizer} ->
          if i == 1, do: IO.puts("0/#{iterate} [                    ]")

          {x_batch, t_batch} =
            Enum.zip(x_train, t_train)
            |> Enum.shuffle()
            |> Enum.take(batch_size)
            |> Enum.unzip()

          grad_model =
            if parallelize == False do
              Broca.Models.TwoLayerNet.gradient(loop_model, loss_layer, x_batch, t_batch)
            else
              Broca.Trainer.parallel_gradient(loop_model, loss_layer, x_batch, t_batch)
            end

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
            if is_nil(test_data) do
              IO.puts(
                "\e[1A#{i}/#{iterate} [#{
                  if progress != 0, do: List.to_string(for _ <- 1..(progress * 2), do: "=")
                }] - loss: #{Float.floor(loss, 5)} - acc: #{Float.floor(acc, 5)}"
              )
            else
              {x_test, t_test} = test_data
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
          end

          {updated_model, updated_optimizer}
        end)

      {epoch_model, epoch_optimizer}
    end)
  end

  def parallel_gradient(model, loss_layer, x_train, t_train) do
    Enum.zip(x_train, t_train)
      |> Flow.from_enumerable(max_demand: 1, stages: 4)
      |> Flow.map(fn {x, t} -> Broca.Models.TwoLayerNet.gradient(model, loss_layer, x, t) end)
      |> Enum.to_list()
      |> Enum.reduce(
        model,
        fn result_model, layers ->
          Enum.zip(layers, result_model)
          |> Enum.map(fn {layer, r} -> Layer.batch_update(layer, r) end)
        end
      )
  end
end