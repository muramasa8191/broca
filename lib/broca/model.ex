defmodule Broca.Models do
  @moduledoc """
  Models
  """

  defmodule Model do
    @type t :: Layer.t()

    defstruct layers: [], loss_layer: nil, optimizer: %Broca.Optimizers.SGD{}

    def init(layers, loss_type, optimizer_type) do
      %Broca.Models.Model{
        layers: layers,
        loss_layer: Broca.Losses.create(loss_type),
        optimizer: Broca.Optimizers.create(optimizer_type, layers)
      }
    end

    def replace_layers({layers, output}, model) do
      {%Broca.Models.Model{model | layers: layers}, output}
    end

    @spec forward([t], [number]) :: [t]
    def forward(model, input) do
      model.layers
      |> Enum.reduce(
        {[], input},
        fn layer, {layers, input} ->
          {layer, out} = Layer.forward(layer, input)
          {[layer] ++ layers, out}
        end
      )
      |> replace_layers(model)
    end

    @spec backward([number], [t]) :: [t]
    def backward(dout, model) do
      model.layers
      |> Enum.reduce(
        {[], dout},
        fn layer, {layers, dout} ->
          {layer, out} = Layer.backward(layer, dout)
          {[layer] ++ layers, out}
        end
      )
      |> replace_layers(model)
    end

    def gradient(model, x, t) do
      {forward_model, _} = forward(model, x)

      Loss.backward(forward_model.loss_layer, t)
      |> Model.backward(forward_model)
      |> elem(0)
    end

    def accuracy(y, t) do
      Enum.zip(Broca.NN.argmax(y), Broca.NN.argmax(t))
      |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)
      |> Kernel./(length(t))
    end

    def loss_and_accuracy(model, {x, t}, parallel \\ 1) do
      chunk_unit = div(length(x), parallel)
      Enum.zip(x, t)
      |> Stream.chunk_every(chunk_unit)
      |> Flow.from_enumerable(max_demand: 1, stages: 4)
      |> Flow.map(fn chunk ->
        {x_chunk, t_chunk} = Enum.unzip(chunk)
        {_, y} = forward(model, x_chunk)

        [
          Loss.loss(model.loss_layer, y, t_chunk) * chunk_unit,
          Enum.zip(Broca.NN.argmax(y), Broca.NN.argmax(t_chunk))
          |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)
        ]
      end)
      |> Enum.reduce(
        [0.0, 0.0],
        fn loss_accurary, acc -> Broca.NN.add(acc, loss_accurary) end
      )
      |> Enum.reduce({}, &Tuple.append(&2, &1 / length(t)))
    end

    def update(model, learning_rate \\ 0.1)

    def update(models, learning_rate) when is_list(models) do
      s = System.os_time(:millisecond)

      {{result_layers, result_optimizer}, _} =
        models
        |> Enum.reduce(
          {{hd(models).layers, hd(models).optimizer}, 0},
          fn new_model, {{updated_model, updated_optimizer}, cnt} ->
            {
              Broca.Optimizer.batch_update(
                updated_optimizer,
                updated_model,
                new_model.layers,
                learning_rate,
                cnt
              ),
              cnt + 1
            }
          end
        )

      IO.puts("* Model update: #{System.os_time(:millisecond) - s}msec")

      %Broca.Models.Model{hd(models) | layers: result_layers, optimizer: result_optimizer}
    end

    def update(model, learning_rate) do
      {result_layers, result_optimizer} =
        Broca.Optimizer.update(model.optimizer, model.layers, learning_rate)

      %Broca.Models.Model{model | layers: result_layers, optimizer: result_optimizer}
    end

    def loss(model, {x, t}) do
      {_, y} = forward(model, x)
      {Loss.loss(model.loss_layer, y, t), y}
    end
  end

  defmodule TwoLayerNet do
    def new(input_size, hidden_size, out_size) do
      affine1 =
        Broca.Layers.Affine.new(
          Broca.Random.randn(input_size, hidden_size)
          |> Broca.NN.mult(:math.sqrt(2.0 / input_size)),
          List.duplicate(0.0, hidden_size),
          :relu
        )

      affine2 =
        Broca.Layers.Affine.new(
          Broca.Random.randn(hidden_size, out_size)
          |> Broca.NN.mult(:math.sqrt(2.0 / hidden_size)),
          List.duplicate(0.0, out_size),
          :softmax
        )

      {[
         affine1,
         affine2
       ], %Broca.Losses.CrossEntropyError{}}
    end
  end

  defmodule SimpleConvolutionNet do
    def new(input_size, filter_size, hidden_size, output_size) do
      Broca.Models.Model.init(
        [
          Broca.Layers.Convolution.new(5, 5, 1, filter_size, 1, 0, :relu),
          Broca.Layers.MaxPooling.new(2, 2, 2, 0),
          Broca.Layers.Affine.new(input_size, hidden_size, :relu),
          Broca.Layers.Affine.new(hidden_size, output_size, :softmax)
        ],
        :cross_entropy_error,
        :adam
      )
    end
  end

  defmodule ConvNet do
    def new() do
      Broca.Models.Model.init(
        [
          Broca.Layers.Convolution.new(3, 3, 3, 32, 1, 1, :relu),
          Broca.Layers.Convolution.new(3, 3, 32, 32, 1, 1, :relu),
          Broca.Layers.MaxPooling.new(2, 2, 2, 0),
          Broca.Layers.Dropout.new(0.25),
          Broca.Layers.Convolution.new(3, 3, 32, 64, 1, 1, :relu),
          Broca.Layers.Convolution.new(3, 3, 64, 64, 1, 1, :relu),
          Broca.Layers.MaxPooling.new(2, 2, 2, 0),
          Broca.Layers.Dropout.new(0.25),
          Broca.Layers.Affine.new(4096, 512, :relu),
          Broca.Layers.Dropout.new(0.5),
          Broca.Layers.Affine.new(512, 10, :softmax)
        ],
        :cross_entropy_error,
        :adam
      )
    end
  end
end
