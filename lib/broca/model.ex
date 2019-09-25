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

      {outs, res}
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
  end

  defmodule TwoLayerNet do
    def new(input_size, hidden_size, out_size, weight_init_std \\ 0.01) do
      affine1 =
        Broca.Layers.Affine.new(
          "a1",
          Broca.Random.randn(input_size, hidden_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, hidden_size)
        )

      affine2 =
        Broca.Layers.Affine.new(
          "a2",
          Broca.Random.randn(hidden_size, out_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, out_size)
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

    def accuracy(model, x, t) do
      {_, y} = predict(model, x)
      pred = Broca.NN.argmax(y)
      ans = Broca.NN.argmax(t)

      sum =
        Enum.zip(pred, ans)
        |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)

      sum / length(pred)
    end

    def gradient(model, loss_layer, t) do
      {new_model, _} =
        Loss.backward(loss_layer, t)
        |> Model.backward(model)

      new_model
    end

    def numerical_gradient(model, loss_layer, x, t) do
      base_func = fn name, idx1, idx2, idx3, diff ->
        y =
          model
          |> Enum.reduce(x, fn layer, out ->
            {_, res} = Layer.gradient_forward(layer, out, name, idx1, idx2, idx3, diff)
            res
          end)

        Loss.loss(loss_layer, y, t)
      end

      [a1, r, a2, s] = model

      {a1grads, a2grads} =
        1..4
        |> Enum.to_list()
        |> Flow.from_enumerable(max_demand: 1, stages: 4)
        |> Flow.map(
          &case &1 do
            1 ->
              IO.puts("dw1 start")

              {:dw1,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx1, idx2, diff -> base_func.("a1", idx1, idx2, -1, diff) end,
                 a1.params,
                 :weight
               )}

            2 ->
              IO.puts("db1 start")

              {:db1,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx, diff -> base_func.("a1", -1, -1, idx, diff) end,
                 a1.params,
                 :bias
               )}

            3 ->
              IO.puts("dw2 start")

              {:dw2,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx1, idx2, diff -> base_func.("a2", idx1, idx2, -1, diff) end,
                 a2.params,
                 :weight
               )}

            4 ->
              IO.puts("db2 start")

              {:db2,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx, diff -> base_func.("a2", -1, -1, idx, diff) end,
                 a2.params,
                 :bias
               )}
          end
        )
        |> Enum.reduce({[], []}, fn {key, list}, {a1grads, a2grads} ->
          case key do
            :dw1 -> Keyword.put_new(a1grads, key, list)
            :db1 -> Keyword.put_new(a1grads, key, list)
            :dw2 -> Keyword.put_new(a2grads, key, list)
            :db2 -> Keyword.put_new(a2grads, key, list)
          end
        end)

      IO.puts("numerical_gradient done")
      [%Broca.Layers.Affine{a1 | grads: a1grads}, r, %Broca.Layers.Affine{a2 | grads: a2grads}, s]
    end

    def loss(model, loss_layer, x, t) do
      {_, y} = predict(model, x)
      Loss.loss(loss_layer, y, t)
    end

    def update(model, optimizer, learning_rate \\ 0.1) do
      Optimizer.update(optimizer, model, learning_rate)
    end
  end

  defmodule TwoLayerNet2 do
    def new(input_size, hidden_size, out_size, weight_init_std \\ 0.01) do
      affine1 =
        Broca.Layers.Affine.new(
          "a1",
          Broca.Random.randn(input_size, hidden_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, hidden_size)
        )

      affine2 =
        Broca.Layers.Affine.new(
          "a2",
          Broca.Random.randn(hidden_size, out_size) |> Broca.NN.mult(weight_init_std),
          List.duplicate(0.0, out_size)
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

    def accuracy(model, x, t) do
      {_, y} = predict(model, x)
      pred = Broca.NN.argmax(y)
      ans = Broca.NN.argmax(t)

      sum =
        Enum.zip(pred, ans)
        |> Enum.reduce(0, fn {p, a}, acc -> if p == a, do: acc + 1, else: acc end)

      sum / length(pred)
    end

    def loss(model, loss_layer, x, t) do
      {_, y} = predict(model, x)
      Loss.loss(loss_layer, y, t)
    end

    def gradient(model, loss_layer, x, t) do
      {forward_model, _} = predict(model, x)

      {backward_model, _} =
        Loss.backward(loss_layer, t)
        |> Model.backward(forward_model)

      backward_model
    end

    def numerical_gradient(model, loss_layer, x, t) do
      base_func = fn name, idx1, idx2, idx3, diff ->
        y =
          model
          |> Enum.reduce(x, &Layer.gradient_forward(&1, &2, name, idx1, idx2, idx3, diff))

        Loss.loss(loss_layer, y, t)
      end

      [a1, r, a2, s] = model

      dw1 =
        Broca.NumericalGradient.numerical_gradient(
          fn idx1, idx2, diff -> base_func.("a1", idx1, idx2, -1, diff) end,
          a1.params,
          :weight
        )

      db1 =
        Broca.NumericalGradient.numerical_gradient(
          fn idx, diff -> base_func.("a1", -1, -1, idx, diff) end,
          a1.params,
          :bias
        )

      dw2 =
        Broca.NumericalGradient.numerical_gradient(
          fn idx1, idx2, diff -> base_func.("a2", idx1, idx2, -1, diff) end,
          a2.params,
          :weight
        )

      db2 =
        Broca.NumericalGradient.numerical_gradient(
          fn idx, diff -> base_func.("a2", -1, -1, idx, diff) end,
          a2.params,
          :bias
        )

      [
        %Broca.Layers.Affine{a1 | grads: [weight: dw1, bias: db1]},
        r,
        %Broca.Layers.Affine{a2 | grads: [weight: dw2, bias: db2]},
        s
      ]
    end

    def update(model, optimizer, learning_rate \\ 0.1) do
      Optimizer.update(optimizer, model, learning_rate)
    end
  end
end
