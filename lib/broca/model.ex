defmodule Broca.Models do
  @moduledoc """
  Models
  """

  defmodule Model do
    @type t :: Layer.t()

    @spec forward([t], [number]) :: [t]
    def forward(layers, input) do
      {outs, res} =
        layers
        |> Enum.reduce(
          {[], input},
          fn layer, {layers, input} ->
            {layer, out} = Layer.forward(layer, input)
            {[layer] ++ layers, out}
          end
        )

      {Enum.reverse(outs), res}
    end

    @spec backward([number], [t]) :: [t]
    def backward(dout, layers) do
      {outs, res} =
        Enum.reverse(layers)
        |> Enum.reduce(
          {[], dout},
          fn layer, {layers, dout} ->
            {layer, out} = Layer.backward(layer, dout)
            {[layer] ++ layers, out}
          end
        )

      {outs, res}
    end
  end

  defmodule TwoLayerNet do
    def new(input_size, hidden_size, out_size, weight_init_std \\ 0.01) do
      affine1 =
        Broca.Layers.Affine.new(
          Broca.Random.randn(input_size, hidden_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, hidden_size)
        )

      affine2 =
        Broca.Layers.Affine.new(
          Broca.Random.randn(hidden_size, out_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, hidden_size)
        )

      {[
         affine1,
         %Broca.Activations.ReLU{},
         affine2,
         %Broca.Activations.Softmax{}
       ], %Broca.Losses.CrossEntropyError{}}
    end

    def predict(model, x) do
      Model.forward(model, x)
    end

    def accuracy(y, t) do
      pred = Broca.NN.argmax(y)
      ans = Broca.NN.argmax(t)

      sum =
        Enum.zip(pred, ans)
        |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)

      sum / length(pred)
    end

    def gradient(model, loss_layer, x, t) do
      loss = Loss.forward(loss_layer, x, t)
      IO.inspect(loss)

      {new_model, _} =
        Loss.backward(loss_layer, t)
        |> Model.backward(model)

      new_model
    end

    def update(model, learning_rate \\ 0.1) do
      model
      |> Enum.map(&Layer.update(&1, learning_rate))
    end
  end
end
