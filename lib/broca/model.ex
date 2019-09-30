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
          # IO.puts("new dout")
          # IO.inspect(dout)
          {[layer] ++ layers, out}
        end
      )
    end
  end

  defmodule TwoLayerNet do
    # def new(input_size, hidden_size, out_size, weight_init_std \\ 0.01) do
    def new(input_size, hidden_size, out_size) do
      affine1 =
        Broca.Layers.Affine.new(
          "a1",
          Broca.Random.randn(input_size, hidden_size) |> Broca.NN.mult(:math.sqrt(2.0 / input_size)),
          List.duplicate(0.0, hidden_size)
        )

      affine2 =
        Broca.Layers.Affine.new(
          "a2",
          Broca.Random.randn(hidden_size, out_size) |> Broca.NN.mult(:math.sqrt(2.0 / hidden_size)),
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

    def gradient(model, loss_layer, x, t) do
      {forward_model, _} = predict(model, x)
      # Loss.loss(loss_layer, y, t) |> IO.inspect
      # IO.puts("backward start")
      {backward_model, _} =
        Loss.backward(loss_layer, t)
        # |> IO.inspect
        |> Model.backward(forward_model)

      backward_model
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
        |> Flow.from_enumerable(max_demand: 1, stages: 4)
        |> Flow.map(
          &case &1 do
            1 ->
              {:dw1,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx1, idx2, diff -> base_func.("a1", idx1, idx2, -1, diff) end,
                 a1.params,
                 :weight
               )}

            2 ->
              {:db1,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx, diff -> base_func.("a1", -1, -1, idx, diff) end,
                 a1.params,
                 :bias
               )}

            3 ->
              {:dw2,
               Broca.NumericalGradient.numerical_gradient(
                 fn idx1, idx2, diff -> base_func.("a2", idx1, idx2, -1, diff) end,
                 a2.params,
                 :weight
               )}

            4 ->
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
            :dw1 ->
              a1grads = Keyword.put_new(a1grads, :weight, list)
              {a1grads, a2grads}

            :db1 ->
              a1grads = Keyword.put_new(a1grads, :bias, list)
              {a1grads, a2grads}

            :dw2 ->
              a2grads = Keyword.put_new(a2grads, :weight, list)
              {a1grads, a2grads}

            :db2 ->
              a2grads = Keyword.put_new(a2grads, :bias, list)
              {a1grads, a2grads}
          end
        end)

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
    # def new(input_size, hidden_size, out_size, weight_init_std \\ 0.01) do
    def new(input_size, hidden_size, out_size) do
      affine1 =
        Broca.Layers.Affine.new(
          "a1",
          Broca.Random.randn(input_size, hidden_size) |> Broca.NN.mult(:math.sqrt(2.0 / input_size)),
          List.duplicate(0.0, hidden_size)
        )

      affine2 =
        Broca.Layers.Affine.new(
          "a2",
          Broca.Random.randn(hidden_size, out_size) |> Broca.NN.mult(:math.sqrt(2.0 / hidden_size)),
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
        |> IO.inspect
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
