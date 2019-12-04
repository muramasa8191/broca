defmodule Broca.Trainer do
  defmodule Setting do
    defstruct epochs: 1, iterates: 1, batch_size: 1, learning_rate: 0.001, parallel: 1

    def new(epochs, iterates, batch_size, learning_rate, parallel) do
      %Broca.Trainer.Setting{
        epochs: epochs,
        iterates: iterates,
        batch_size: batch_size,
        learning_rate: learning_rate,
        parallel: parallel
      }
    end
  end

  def gradient({x_train, t_train}, model, 1) do
    Broca.Models.Model.gradient(model, x_train, t_train)
  end

  def gradient({x_train, t_train}, model, parallel_size) do
    Enum.zip(x_train, t_train)
    |> Enum.chunk_every(div(length(x_train), parallel_size))
    |> Flow.from_enumerable(max_demand: 1, stages: 4)
    |> Flow.map(fn data ->
      {x, t} = Enum.unzip(data)
      Broca.Models.Model.gradient(model, x, t)
    end)
    |> Enum.to_list()
  end

  defp time_format(time_integer) do
    time_string = Integer.to_string(time_integer)
    if String.length(time_string) > 1, do: time_string, else: "0" <> time_string
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

  defp choose_random_data({x_train, t_train}, batch_size) do
    Enum.zip(x_train, t_train)
    |> Enum.take_random(batch_size)
    |> Enum.unzip()
  end

  defp print_start_log(train_data, nil) do
    IO.puts("Train on #{length(elem(train_data, 0))} samples.")
  end

  defp print_start_log(train_data, test_data) do
    IO.puts(
      "Train on #{length(elem(train_data, 0))} samples, Validation on #{
        length(elem(test_data, 0))
      } samples."
    )
  end

  defp print_report(train_result, progress, start_time, validation_result, is_last) do
    display_result(
      train_result,
      progress,
      System.os_time(:second) - start_time,
      is_last,
      validation_result
    )
  end

  defp create_progress(iterates, setting) do
    {
      "#{setting.iterates - iterates}/#{setting.iterates}",
      floor((setting.iterates - iterates) / setting.iterates * 20)
    }
  end

  defp report(model, progress, batch_data, start_time, validation_result, eta, setting) do
    print_report(
      Broca.Models.Model.loss_and_accuracy(model, batch_data, setting.parallel),
      progress,
      start_time,
      validation_result,
      eta
    )

    model
  end

  defp calc_ETA(rest_iterate, total_iterate, start_time) do
    div(System.os_time(:second) - start_time, total_iterate - rest_iterate) * rest_iterate
  end

  defp iterate(model, 0, start_time, batch_data, test_data, setting) do
    report(
      model,
      create_progress(0, setting),
      batch_data,
      start_time,
      Broca.Models.Model.loss_and_accuracy(model, test_data, setting.parallel),
      nil,
      setting
    )
  end

  defp iterate(model, rest_iterate, start_time, train_data, test_data, setting) do
    if rest_iterate == setting.iterates,
      do: IO.puts("0/#{setting.iterates} [                    ]")

    batch_data = choose_random_data(train_data, setting.batch_size)

    batch_data
    |> gradient(model, setting.parallel)
    |> Broca.Models.Model.update(setting.learning_rate)
    |> report(
      create_progress(rest_iterate - 1, setting),
      batch_data,
      start_time,
      nil,
      calc_ETA(rest_iterate - 1, setting.iterates, start_time),
      setting
    )
    |> iterate(rest_iterate - 1, start_time, train_data, test_data, setting)
  end

  defp epoch(0, model, _) do
    model
  end

  defp epoch(rest_epoch, model, train_data, test_data, setting) do
    IO.puts("Epoch #{setting.epochs - rest_epoch + 1}/#{setting.epochs}")

    epoch(
      rest_epoch - 1,
      iterate(model, setting.iterates, System.os_time(:second), train_data, test_data, setting),
      setting
    )
  end

  def train(model, setting, train_data, test_data \\ nil) do
    print_start_log(train_data, test_data)
    epoch(setting.epochs, model, train_data, test_data, setting)
  end

  def display_result({loss, accuracy}, {iterate, progress}, start_time, nil, nil) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}] - #{
        System.os_time(:second) - start_time
      }s - loss: #{Float.floor(loss, 5)} - acc: #{Float.floor(accuracy, 5)}"
    )
  end

  def display_result(
        {loss, accuracy},
        {iterate, progress},
        start_time,
        nil,
        {test_loss, test_accuracy}
      ) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}] - #{
        System.os_time(:second) - start_time
      }s - loss: #{Float.floor(loss, 5)} - acc: #{Float.floor(accuracy, 5)} - val_loss: #{
        Float.floor(test_loss, 5)
      } - val_acc: #{Float.floor(test_accuracy, 5)}"
    )
  end

  def display_result({loss, accuracy}, {iterate, progress}, _, rest, _) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}#{
        List.to_string(for _ <- 1..(20 - progress), do: " ")
      }] - ETA: #{time_string(rest)} - loss: #{Float.floor(loss, 5)} - acc: #{
        Float.floor(accuracy, 5)
      }                          "
    )
  end
end
