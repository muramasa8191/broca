defmodule Broca.Trainer do
  alias Broca.{
    Layers,
    Loss,
    Optimizers,
    Trainer,
    Nif.NN,
    DataLoader
  }

  require Logger

  def setting(epochs, batch_size, optimizer_type, learning_rate, pallarel_size) do
    [
      epochs: epochs,
      batch_size: batch_size,
      optimizer_type: optimizer_type,
      learning_rate: learning_rate,
      pallarel_size: pallarel_size
    ]
  end

  defp print_start_log(train_data_loader, nil) do
    IO.puts("Train on #{DataLoader.size(train_data_loader)} samples.")
  end

  defp print_start_log(train_data_loader, test_data_loader) do
    IO.puts(
      "Train on #{DataLoader.size(train_data_loader)} samples, Validation on #{
        DataLoader.size(test_data_loader)
      } samples."
    )
  end

  def train(model, setting, train_data_loader, test_data_loader) do
    print_start_log(train_data_loader, test_data_loader)

    setting =
      Keyword.put_new(
        setting,
        :iterates,
        div(DataLoader.size(train_data_loader), setting[:batch_size])
      )

    1..setting[:epochs]
    |> Enum.reduce(
      Layers.set_optimizers(model, setting[:optimizer_type], setting[:learning_rate]),
      fn current_epoch, model_ ->
        task =
          Task.async(Broca.Trainer, :epoch, [
            current_epoch,
            model_,
            setting,
            train_data_loader,
            test_data_loader
          ])

        Task.await(task, :infinity)
      end
    )
  end

  defp accuracy(y, t) do
    Enum.zip(NN.argmax(y), NN.argmax(t))
    |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)
    |> Kernel./(length(t))
  end

  def forward(input, model) do
    Logger.debug("forward start")

    model
    |> Enum.reduce(
      {input, []},
      fn layer, {x, new_model} ->
        case layer do
          # Convolution
          [layer_type, params, optimizer, fix_params] ->
            task = Task.async(Layers, :forward, [x, [layer_type, params, fix_params]])
            {out, config} = Task.await(task, :infinity)
            {out, [[layer_type, params, optimizer, fix_params, config]] ++ new_model}

          # Affine
          [layer_type, params, optimizer] ->
            task = Task.async(Layers, :forward, [x, [layer_type, params]])
            {out, config} = Task.await(task, :infinity)
            {out, [[layer_type, params, optimizer, config]] ++ new_model}

          # MaxPooling, Dropout
          [layer_type, fix_params] ->
            task = Task.async(Layers, :forward, [x, [layer_type, fix_params]])
            {out, config} = Task.await(task, :infinity)
            {out, [[layer_type, fix_params, config]] ++ new_model}

          # Activation(ReLU, Softmax)
          [layer_type] ->
            task = Task.async(Layers, :forward, [x, layer_type])
            {out, config} = Task.await(task, :infinity)
            {out, [[layer_type, config]] ++ new_model}
        end
      end
    )
  end

  def backward(dout, model) do
    Logger.debug("backward start")

    model
    |> Enum.reduce(
      {dout, []},
      fn layer, {input, new_model} ->
        case layer do
          # Convolution
          [layer_type, params, optimizer, fix_params, config] ->
            task =
              Task.async(Layers, :backward, [input, [layer_type, params, fix_params, config]])

            {res, grads} = Task.await(task, :infinity)
            {res, [[layer_type, params, optimizer, fix_params, grads]] ++ new_model}

          # Affine
          [layer_type, params, optimizer, config] ->
            task = Task.async(Layers, :backward, [input, [layer_type, params, config]])
            {res, grads} = Task.await(task, :infinity)
            {res, [[layer_type, params, optimizer, grads]] ++ new_model}

          # MaxPooling, Dropout
          [layer_type, fix_params, config] ->
            task = Task.async(Layers, :backward, [input, [layer_type, fix_params, config]])
            res = Task.await(task, :infinity)
            {res, [[layer_type, fix_params]] ++ new_model}

          # ReLU, Softmax
          [layer_type, config] ->
            task = Task.async(Layers, :backward, [input, [layer_type, config]])
            res = Task.await(task, :infinity)
            {res, [[layer_type]] ++ new_model}
        end
      end
    )
  end

  defp gradient_impl(batch_data, model) do
    {x_batch, t_batch} = Enum.unzip(batch_data)

    {out, forward_model} = forward(x_batch, model)
    loss = Loss.loss(:cross_entropy_error, out, t_batch)
    acc = accuracy(out, t_batch)

    {_, backward_model} = backward(t_batch, forward_model)

    {backward_model, loss, acc}
  end

  def update(models) when is_list(hd(hd(models))) do
    Logger.debug("update start\n")

    models
    |> Enum.reduce(
      {hd(models), 0},
      fn new_model, {prev_model, idx} ->
        updated_model =
          Enum.zip(new_model, prev_model)
          |> Enum.map(fn {layer, prev_layer} ->
            case layer do
              # Convolution
              [layer_type, _, _, fix_params, grads] ->
                [_, params, [opt_type, opt_config] | _] = prev_layer

                task =
                  Task.async(Optimizers, :optimize, [
                    opt_type,
                    params,
                    grads,
                    opt_config,
                    idx == 0
                  ])

                {new_params, new_config} = Task.await(task, :infinity)

                [layer_type, new_params, [opt_type, new_config], fix_params]

              # Affine
              [layer_type, _, _, grads] ->
                [_, params, [opt_type, opt_config] | _] = prev_layer

                task =
                  Task.async(Optimizers, :optimize, [
                    opt_type,
                    params,
                    grads,
                    opt_config,
                    idx == 0
                  ])

                {params, new_config} = Task.await(task, :infinity)

                [layer_type, params, [opt_type, new_config]]

              # ReLU, Softmax, MaxPooling, Dropout
              _ ->
                layer
            end
          end)

        {updated_model, idx + 1}
      end
    )
    |> elem(0)
  end

  def update(model) do
    Enum.map(
      model,
      fn {layer, prev_layer} ->
        case layer do
          [layer_type, _, _, grads] ->
            [_, params, [opt_type, opt_config]] = prev_layer
            task = Task.async(Optimizers, :optimize, [opt_type, params, grads, opt_config])
            {new_params, new_config} = Task.await(task, :infinity)

            [layer_type, new_params, [opt_type, new_config]]

          [layer_type] ->
            [layer_type]
        end
      end
    )
  end

  # def gradient(batch_data, model, 1, _) do
  #   gradient_impl(batch_data, model)
  # end

  def gradient(batch_data, model, parallel, batch_size) do
    {grad_model, accum_loss, accum_accuracy} =
      batch_data
      |> Stream.chunk_every(div(batch_size, parallel))
      |> Flow.from_enumerable(max_demand: 1, stages: 4)
      |> Flow.map(&gradient_impl(&1, model))
      |> Enum.reduce(
        {[], 0.0, 0.0},
        fn {sub_model, l, a}, {acc_model, acc_loss, acc_accuracy} ->
          {[sub_model] ++ acc_model, acc_loss + l, acc_accuracy + a}
        end
      )

    {grad_model, accum_loss / parallel, accum_accuracy / parallel}
  end

  def get_batch(train_data_loader, batch_size) do
    DataLoader.get_batch_data(train_data_loader, batch_size)
  end

  def iterate(model, train_data_loader, setting) do
    task = Task.async(Trainer, :get_batch, [train_data_loader, setting[:batch_size]])

    task2 =
      Task.async(Trainer, :gradient, [
        Task.await(task, :infinity),
        model,
        setting[:pallarel_size],
        setting[:batch_size]
      ])

    {backward_model, loss, acc} = Task.await(task2, :infinity)

    {update(backward_model), loss, acc}
  end

  def validation(model, validation_data_loader, setting) do
    1..div(DataLoader.size(validation_data_loader), setting[:batch_size])
    |> Enum.reduce(
      {[0.0, 0.0], validation_data_loader},
      fn _, {[accum_loss, accum_accuracy], dataloader} ->
        {data, loader} = DataLoader.get_batch_data(dataloader, setting[:batch_size])
        {x_batch, t_batch} = Enum.unzip(data)
        {out, _} = forward(x_batch, model)
        loss = Loss.loss(:cross_entropy_error, out, t_batch) * setting[:batch_size]
        acc = accuracy(out, t_batch) * setting[:batch_size]
        {[accum_loss + loss, accum_accuracy + acc], loader}
      end
    )
    |> elem(0)
    |> Enum.reduce({}, &Tuple.append(&2, &1 / DataLoader.size(validation_data_loader)))
  end

  def epoch(current_epoch, model, setting, train_data_loader, test_data_loader) do
    IO.puts("Epoch #{current_epoch}/#{setting[:epochs]}")
    IO.puts("0/#{setting[:iterates]} [                    ]")
    start_sec = System.os_time(:second)

    new_model =
      1..setting[:iterates]
      |> Enum.reduce(
        model,
        fn i, model_ ->
          start_sec = System.os_time(:second)
          task = Task.async(Broca.Trainer, :iterate, [model_, train_data_loader, setting])
          {model_, loss, acc} = Task.await(task, :infinity)

          display_result(
            {loss, acc},
            create_progress(i, setting),
            nil,
            calc_ETA(setting[:iterates] - i, System.os_time(:second) - start_sec),
            nil
          )

          model_
        end
      )

    task = Task.async(Trainer, :validation, [new_model, test_data_loader, setting])
    validation_result = Task.await(task, :infinity)

    display_result(
      create_progress(setting[:iterates], setting),
      start_sec,
      validation_result
    )

    new_model
  end

  defp create_progress(iterates, setting) do
    {
      "#{iterates}/#{setting[:iterates]}",
      floor(iterates / setting[:iterates] * 20)
    }
  end

  defp calc_ETA(rest_iterate, duration) do
    duration * rest_iterate
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

  def display_result({iterate, progress}, start_time, nil) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}] - #{
        System.os_time(:second) - start_time
      }s                                            "
    )
  end

  def display_result(
        {iterate, progress},
        start_time,
        {test_loss, test_accuracy}
      ) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}] - #{
        time_string(System.os_time(:second) - start_time)
      } - val_loss: #{Float.floor(test_loss, 5)} - val_acc: #{Float.floor(test_accuracy, 5)}                    "
    )
  end

  def display_result({loss, accuracy}, {iterate, progress}, _, rest, _) do
    IO.puts(
      "\e[1A#{iterate} [#{if progress != 0, do: List.to_string(for _ <- 1..progress, do: "=")}#{
        if progress != 20, do: List.to_string(for _ <- 1..(20 - progress), do: " ")
      }] - ETA: #{time_string(rest)} - loss: #{Float.floor(loss, 5)} - acc: #{
        Float.floor(accuracy, 5)
      }                          "
    )
  end
end
