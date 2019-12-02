defmodule Broca.Models do
  @moduledoc """
  Models
  """

  defmodule Model do
    @type t :: Layer.t()

    @spec forward([t], [number]) :: [t]
    def forward(layers, input) do
      layers
      |> Enum.reduce(
        {[], input},
        fn layer, {layers, input} ->
          {layer, out} = Layer.forward(layer, input)
          {[layer] ++ layers, out}
        end
      )
    end

    @spec backward([number], [t]) :: [t]
    def backward(dout, layers) do
      layers
      |> Enum.reduce(
        {[], dout},
        fn layer, {layers, dout} ->
          {layer, out} = Layer.backward(layer, dout)
          {[layer] ++ layers, out}
        end
      )
    end

    def gradient(model, loss_layer, x, t) do
      {forward_model, _} = forward(model, x)

      Loss.backward(loss_layer, t)
      |> Model.backward(forward_model)
      |> elem(0)
    end

    def accuracy(y, t) do
      Enum.zip(Broca.NN.argmax(y), Broca.NN.argmax(t))
      |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)
      |> Kernel./(length(t))
    end

    def loss_and_accuracy(model, loss_layer, {x, t}, chunk_amount \\ 1) do
      Enum.zip(x, t)
      |> Stream.chunk_every(chunk_amount)
      |> Flow.from_enumerable(max_demand: 1, stages: 4)
      |> Flow.map(fn chunk ->
        {x_chunk, t_chunk} = Enum.unzip(chunk)
        {_, y} = forward(model, x_chunk)

        [
          Loss.loss(loss_layer, y, t_chunk) * chunk_amount,
          Enum.zip(Broca.NN.argmax(y), Broca.NN.argmax(t_chunk))
          |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)
        ]
      end)
      |> Enum.reduce(
        [0.0, 0.0],
        fn loss_accurary, acc -> Broca.NN.add(acc, loss_accurary) end
      )
      |> Broca.NN.division(length(t))
    end

    def update(model, optimizer, learning_rate \\ 0.1)

    def update(models, optimizer, learning_rate) when is_list(hd(models)) do
      models
      |> Enum.reduce(
        {{hd(models), optimizer}, 0},
        fn new_model, {{updated_model, updated_optimizer}, cnt} ->
          {
            Broca.Optimizer.batch_update(
              updated_optimizer,
              updated_model,
              new_model,
              learning_rate,
              cnt
            ),
            cnt + 1
          }
        end
      )
      |> elem(0)
    end

    def update(model, optimizer, learning_rate) do
      Broca.Optimizer.update(optimizer, model, learning_rate)
    end

    def loss(model, loss_layer, {x, t}) do
      {_, y} = forward(model, x)
      {Loss.loss(loss_layer, y, t), y}
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

    def predict(model, x) do
      Model.forward(model, x)
    end

    def accuracy(model, x, t) do
      {_, y} = predict(model, x)
      pred = Broca.NN.argmax(y)
      ans = Broca.NN.argmax(t)

      sum =
        Enum.zip(pred, ans)
        |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)

      sum / length(pred)
    end

    def gradient(model, loss_layer, x, t) do
      {forward_model, _} = predict(model, x)

      {backward_model, _} =
        Loss.backward(loss_layer, t)
        |> Model.backward(forward_model)

      backward_model
    end

    def loss(model, loss_layer, x, t) do
      {_, y} = predict(model, x)
      Loss.loss(loss_layer, y, t)
    end

    def update(model, optimizer, learning_rate \\ 0.1) do
      Broca.Optimizer.update(optimizer, model, learning_rate)
    end
  end

  defmodule SimpleConvolutionNet do
    def new(input_size, filter_size, hidden_size, output_size) do
      {
        [
          Broca.Layers.Convolution.new(5, 5, 1, filter_size, 1, 0, :relu),
          Broca.Layers.MaxPooling.new(2, 2, 2, 0),
          Broca.Layers.Affine.new(input_size, hidden_size, :relu),
          Broca.Layers.Affine.new(hidden_size, output_size, :softmax)
        ],
        %Broca.Losses.CrossEntropyError{}
      }
    end
  end
end
