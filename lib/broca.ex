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
      # [a1, _, a2, _] = model0
      # IO.puts("epoch a1 w: #{hd(hd a1.weight)}, b: #{hd a1.bias}")
      # IO.puts("epoch a2 w: #{hd(hd a2.weight)}, b: #{hd a2.bias}")

      {epoch_model, epoch_optimizer} =
        1..iterate
        |> Enum.reduce({e_model, e_optimizer}, fn i, {loop_model, loop_optimizer} ->
          if i == 1, do: IO.puts("0/#{iterate} [                    ]")

          {x_batch, t_batch} =
            zip_data
            |> Enum.shuffle()
            |> Enum.take(batch_size)
            |> Enum.unzip()

          # IO.puts("x_batch size")
          # IO.inspect(Broca.NN.shape(x_batch))
          # [a1, _, a2, _] = loop_model
          # IO.puts("w1:")
          # IO.inspect(hd a1.weight |> Enum.take(10))
          # IO.puts("b1:") 
          # IO.inspect(a1.bias |> Enum.take(10))
          # IO.puts("w2:")
          # IO.inspect(hd a2.weight |> Enum.take(10))
          # IO.puts("b2:")
          # IO.inspect(a2.bias |> Enum.take(10))

          # grad_model =
          #   Broca.Models.TwoLayerNet.numerical_gradient(loop_model, loss_layer, x_batch, t_batch)

          grad_model =
            Broca.Models.TwoLayerNet.gradient(loop_model, loss_layer, x_batch, t_batch)

          # [a11, _, a21, _] = grad_model
          # [a12, _, a22, _] = grad_model2

          # dw1_diff =
          #   Enum.zip(a11.grads[:weight], a12.grads[:weight])
          #   |> Enum.map(fn {ws1, ws2} ->
          #     Enum.zip(ws1, ws2)
          #     |> Enum.map(fn {w1, w2} -> abs(w1 - w2) end)
          #     |> Enum.sum()
          #   end)
          #   |> Enum.sum()
          #   |> Kernel./(length(a11.grads[:weight]) * length(hd(a11.grads[:weight])))

          # db1_diff =
          #   Enum.zip(a11.grads[:bias], a12.grads[:bias])
          #   |> Enum.map(fn {w1, w2} -> abs(w1 - w2) end)
          #   |> Enum.sum()
          #   |> Kernel./(length(a11.grads[:bias]))

          # dw2_diff =
          #   Enum.zip(a21.grads[:weight], a22.grads[:weight])
          #   |> Enum.map(fn {ws1, ws2} ->
          #     Enum.zip(ws1, ws2)
          #     |> Enum.map(fn {w1, w2} -> abs(w1 - w2) end)
          #     |> Enum.sum()
          #   end)
          #   |> Enum.sum()
          #   |> Kernel./(length(a21.grads[:weight]) * length(hd(a21.grads[:weight])))

          # db2_diff =
          #   Enum.zip(a21.grads[:bias], a22.grads[:bias])
          #   |> Enum.map(fn {w1, w2} -> abs(w1 - w2) end)
          #   |> Enum.sum()
          #   |> Kernel./(length(a21.grads[:bias]))

          # IO.puts(
          #   "** dw1_diff: #{dw1_diff}, db1_diff: #{db1_diff}, dw2_diff: #{dw2_diff}, db2_diff: #{
          #     db2_diff
          #   }\n\n"
          # )

          # [a1, _, a2, _] = grad_model
          # IO.puts("grad_w1:")
          # IO.inspect(hd a1.dw |> Enum.take(10))
          # IO.puts("grad_b1:") 
          # IO.inspect(a1.db |> Enum.take(10))
          # IO.puts("grad_w2:")
          # IO.inspect(hd a2.dw |> Enum.take(10))
          # IO.puts("grad_b2:")
          # IO.inspect(a2.db |> Enum.take(10))
          # IO.puts("delete")

          {updated_model, updated_optimizer} =
            Broca.Models.TwoLayerNet.update(grad_model, loop_optimizer, learning_rate)

          # [a1, _, a2, _] = updated_model
          # IO.puts("new_w1:")
          # IO.inspect(hd a1.weight |> Enum.take(10))
          # IO.puts("new_b1:") 
          # IO.inspect(a1.bias |> Enum.take(10))
          # IO.puts("new_w2:")
          # IO.inspect(hd a2.weight |> Enum.take(10))
          # IO.puts("new_b2:")
          # IO.inspect(a2.bias |> Enum.take(10))
          # IO.puts("delete")
          # IO.puts("accuracy")
          # [a1, _, a2, _] = updated_model
          # a1.weight |> Broca.NN.shape |> Enum.map(&(IO.puts("#{&1}, ")))
          # a2.weight |> Broca.NN.shape |> Enum.map(&(IO.puts("#{&1}, ")))
          # IO.puts("loss")
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

          # [a1, _, a2, _] = model3
          # IO.puts("last a1 w: #{hd(hd a1.weight)}, b: #{hd a1.bias}")
          # IO.puts("last a2 w: #{hd(hd a2.weight)}, b: #{hd a2.bias}")
          {updated_model, updated_optimizer}
        end)

      {epoch_model, epoch_optimizer}
    end)
  end

  def batch_train(model, loss_layer, x_batch, t_batch) do
    Broca.Models.TwoLayerNet2.gradient(model, loss_layer, x_batch, t_batch)
  end

  def ch05_2(epochs, batch_size, learning_rate \\ 0.1) do
    {x_train, t_train} = Broca.Dataset.MNIST.load_train_data()
    {x_test, t_test} = Broca.Dataset.MNIST.load_test_data()
    {model, loss_layer} = Broca.Models.TwoLayerNet2.new(784, 50, 10)
    data_size = length(x_train)
    IO.puts("Train on #{data_size} samples.")
    zip_data = Enum.zip(x_train, t_train)

    iterate = round(data_size / batch_size) |> max(1)
    optimizer = %Broca.Optimizers.SGD{}

    1..epochs
    |> Enum.reduce({model, optimizer}, fn epoch, {e_model, e_optimizer} ->
      IO.puts("Epoch #{epoch}/#{epochs}")
      # [a1, _, a2, _] = model0
      # IO.puts("epoch a1 w: #{hd(hd a1.weight)}, b: #{hd a1.bias}")
      # IO.puts("epoch a2 w: #{hd(hd a2.weight)}, b: #{hd a2.bias}")

      {epoch_model, epoch_optimizer} =
        1..iterate
        |> Enum.reduce({e_model, e_optimizer}, fn i, {loop_model, loop_optimizer} ->
          if i == 1, do: IO.puts("0/#{iterate} [                    ]")

          shuffle_zip_data =
            zip_data
            |> Enum.shuffle()
            |> Enum.take(batch_size)

          {x_batch, t_batch} =
            shuffle_zip_data
            |> Enum.unzip()

          # IO.puts("x_batch size")
          # IO.inspect(Broca.NN.shape(x_batch))
          # [a1, _, a2, _] = loop_model
          # IO.puts("w1:")
          # IO.inspect(hd a1.weight |> Enum.take(10))
          # IO.puts("b1:") 
          # IO.inspect(a1.bias |> Enum.take(10))
          # IO.puts("w2:")
          # IO.inspect(hd a2.weight |> Enum.take(10))
          # IO.puts("b2:")
          # IO.inspect(a2.bias |> Enum.take(10))

          grad_model =
            shuffle_zip_data
            |> Flow.from_enumerable(max_demand: 4)
            |> Flow.map(fn {x, t} -> batch_train(loop_model, loss_layer, x, t) end)
            |> Enum.to_list()
            # |> IO.inspect
            |> Enum.reduce(
              loop_model,
              fn result_model, mod ->
                Enum.zip(mod, result_model)
                |> Enum.map(fn {m, r} -> Layer.batch_update(m, r) end)
              end
            )

          # [a1, _, a2, _] = grad_model
          # IO.puts("grad_w1:")
          # IO.inspect(hd a1.grads[:weight] |> Enum.take(10))
          # IO.puts("grad_b1:") 
          # IO.inspect(a1.grads[:bias] |> Enum.take(10))
          # IO.puts("grad_w2:")
          # IO.inspect(hd a2.grads[:weight] |> Enum.take(10))
          # IO.puts("grad_b2:")
          # IO.inspect(a2.grads[:bias] |> Enum.take(10))
          # IO.puts("delete")

          {updated_model, updated_optimizer} =
            Broca.Models.TwoLayerNet2.update(grad_model, loop_optimizer, learning_rate)

          # [a1, _, a2, _] = updated_model
          # IO.inspect(a1)
          # IO.puts("new_w1:")
          # IO.inspect(hd a1.params[:weight] |> Enum.take(10))
          # IO.puts("new_b1:") 
          # IO.inspect(a1.params[:bias] |> Enum.take(10))
          # IO.puts("new_w2:")
          # IO.inspect(hd a2.params[:weight] |> Enum.take(10))
          # IO.puts("new_b2:")
          # IO.inspect(a2.params[:bias] |> Enum.take(10))
          # IO.puts("delete")
          # IO.puts("accuracy")
          # [a1, _, a2, _] = updated_model
          # a1.weight |> Broca.NN.shape |> Enum.map(&(IO.puts("#{&1}, ")))
          # a2.weight |> Broca.NN.shape |> Enum.map(&(IO.puts("#{&1}, ")))
          # IO.puts("loss")
          loss = Broca.Models.TwoLayerNet2.loss(updated_model, loss_layer, x_batch, t_batch)
          progress = Float.floor(i / iterate, 1) * 10 |> Kernel.trunc

          if i != iterate do
            IO.puts(
              "\e[1A#{i}/#{iterate} [#{
                if progress != 0, do: List.to_string(for _ <- 1..(progress * 2), do: "=")
              }#{List.to_string(for _ <- 1..((10 - progress) * 2), do: " ")}] - loss: #{
                Float.floor(loss, 5)
              }         "
            )
          else
            acc = Broca.Models.TwoLayerNet2.accuracy(updated_model, x_batch, t_batch)
            test_acc = Broca.Models.TwoLayerNet2.accuracy(updated_model, x_test, t_test)
            test_loss = Broca.Models.TwoLayerNet2.loss(updated_model, loss_layer, x_test, t_test)

            IO.puts(
              "\e[1A#{i}/#{iterate} [#{
                if progress != 0, do: List.to_string(for _ <- 1..(progress * 2), do: "=")
              }] - loss: #{Float.floor(loss, 5)} - acc: #{Float.floor(acc, 5)} - test_loss: #{
                Float.floor(test_loss, 5)
              } - test_acc: #{Float.floor(test_acc, 5)}"
            )
          end

          # [a1, _, a2, _] = model3
          # IO.puts("last a1 w: #{hd(hd a1.weight)}, b: #{hd a1.bias}")
          # IO.puts("last a2 w: #{hd(hd a2.weight)}, b: #{hd a2.bias}")
          {updated_model, updated_optimizer}
        end)

      {epoch_model, epoch_optimizer}
    end)
  end
end
