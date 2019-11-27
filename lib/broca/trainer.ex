defmodule Broca.Trainer do
  def gradient({x_train, t_train}, model, loss_layer, 1) do
    Broca.Models.TwoLayerNet.gradient(model, loss_layer, x_train, t_train)
  end

  def gradient({x_train, t_train}, model, loss_layer, parallel_size) do
    Enum.zip(x_train, t_train)
    |> Enum.chunk_every(div(length(x_train), parallel_size))
    |> Flow.from_enumerable(max_demand: 1, stages: 4)
    |> Flow.map(fn data ->
      {x, t} = Enum.unzip(data)
      Broca.Models.Model.gradient(model, loss_layer, x, t)
    end)
    |> Enum.to_list()
  end

  defp time_format(time_integer) do
    time_string = Integer.to_string(time_integer)
    if String.length(time_string) == 2, do: time_string, else: "0" <> time_string
  end

  defp time_string(time_seconds) do
    hours = div(time_seconds, 3600)
    minutes = rem(div(time_seconds, 60), 60)
    seconds = rem(time_seconds, 60)

    if hours != 0 or minutes != 0 do
      "#{time_format(hours)}:#{time_format(minutes)}:#{time_format(seconds)}"
    else
      "#{seconds}s"
    end
  end

  defp choose_random_data(x_train, t_train, batch_size) do
    Enum.zip(x_train, t_train)
    |> Enum.take_random(batch_size)
    |> Enum.unzip()
  end

  def train(
        model,
        loss_layer,
        optimizer_type,
        {x_train, t_train} = _train_data,
        epochs,
        batch_size,
        learning_rate,
        parallel \\ 1,
        test_data \\ nil
      ) do
    data_size = length(x_train)

    if not is_nil(test_data) do
      IO.puts(
        "Train on #{data_size} samples, Validation on #{length(elem(test_data, 0))} samples."
      )
    else
      IO.puts("Train on #{data_size} samples.")
    end

    iterate = round(data_size / batch_size) |> max(1)

    1..epochs
    |> Enum.reduce({model, Broca.Optimizers.create(optimizer_type, model)}, fn epoch,
                                                                               {e_model,
                                                                                e_optimizer} ->
      IO.puts("Epoch #{epoch}/#{epochs}")
      s = System.os_time(:second)

      1..iterate
      |> Enum.reduce({e_model, e_optimizer}, fn i, {loop_model, loop_optimizer} ->
        if i == 1, do: IO.puts("0/#{iterate} [                    ]")

        batch_data = choose_random_data(x_train, t_train, batch_size)

        {updated_model, updated_optimizer} =
          batch_data
          |> gradient(loop_model, loss_layer, parallel)
          |> Broca.Models.Model.update(loop_optimizer, learning_rate)

        [train_loss, accuracy] =
          Broca.Models.Model.loss_and_accuracy(updated_model, loss_layer, batch_data, div(batch_size, parallel))

        if i != iterate do
          rest = div(System.os_time(:second) - s, i) * (iterate - i)

          Broca.Trainer.display_result(
            train_loss,
            accuracy,
            iterate,
            i,
            floor(i / iterate * 20),
            rest
          )
        else
          Broca.Trainer.display_result(
            train_loss,
            accuracy,
            iterate,
            i,
            floor(i / iterate * 20),
            System.os_time(:second) - s,
            updated_model,
            loss_layer,
            test_data
          )
        end

        {updated_model, updated_optimizer}
      end)
    end)
  end

  def display_result(loss, accuracy, iterate, i, progress, rest) do
    IO.puts(
      "\e[1A#{i}/#{iterate} [#{
        if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")
      }#{List.to_string(for _ <- 1..(20 - progress), do: " ")}] - ETA: #{time_string(rest)} - loss: #{
        Float.floor(loss, 5)
      } - acc: #{Float.floor(accuracy, 5)}                          "
    )
  end

  def display_result(loss, accuracy, iterate, i, progress, duration, _, _, nil) do
    IO.puts(
      "\e[1A#{i}/#{iterate} [#{
        if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")
      }] - #{duration}s - loss: #{
        Float.floor(loss, 5)
      } - acc: #{Float.floor(accuracy, 5)}"
    )
  end

  def display_result(loss, accuracy, iterate, i, progress, duration, model, loss_layer, test_data) do
    [test_loss, test_acc] = Broca.Models.Model.loss_and_accuracy(model, loss_layer, test_data, 20)

    IO.puts(
      "\e[1A#{i}/#{iterate} [#{
        if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")
      }] - #{duration}s - loss: #{
        Float.floor(loss, 5)
      } - acc: #{Float.floor(accuracy, 5)} - val_loss: #{Float.floor(test_loss, 5)} - val_acc: #{
        Float.floor(test_acc, 5)
      }"
    )
  end
end
