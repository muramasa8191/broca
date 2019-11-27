defmodule Broca.Optimizers.Adam do
  defstruct beta1: 0.9, beta2: 0.999, iter: 0, mv: []

  defimpl Inspect, for: Broca.Optimizers.Adam do
    def inspect(adam, _) do
      "Adam: beta1=#{adam.beta1}, beta2=#{adam.beta2}, iter=#{adam.iter}, mv=#{
        Broca.NN.shape_string(adam.mv)
      }"
    end
  end

  defimpl Broca.Optimizer, for: Broca.Optimizers.Adam do
    def init(optimizer, model) do
      mv =
        model
        |> Enum.map(fn layer ->
          if Map.has_key?(layer, :params) do
            Keyword.keys(layer.params)
            |> Enum.reduce(
              Keyword.new(),
              &Keyword.put_new(
                &2,
                &1,
                {Broca.NN.zeros_like(layer.params[&1]), Broca.NN.zeros_like(layer.params[&1])}
              )
            )
          else
            []
          end
        end)

      %Broca.Optimizers.Adam{optimizer | mv: mv}
    end

    def update(adam, model, learning_rate) do
      iter = adam.iter + 1

      lr =
        learning_rate * :math.sqrt(1.0 - :math.pow(adam.beta2, iter)) /
          (1.0 - :math.pow(adam.beta1, iter))

      new_mv = Broca.Optimizers.Adam.update_mv(adam, model)

      updated_model =
        Enum.zip(new_mv, model)
        |> Enum.map(fn {mv, layer} ->
          Layer.update(
            layer,
            fn params, _ ->
              Keyword.keys(params)
              |> Enum.reduce(
                [],
                fn key, keyword ->
                  Keyword.put_new(
                    keyword,
                    key,
                    params[key]
                    |> Broca.NN.subtract(
                      Broca.NN.division(
                        elem(mv[key], 0),
                        Broca.NN.for_each(
                          elem(mv[key], 1),
                          fn val -> :math.sqrt(val) + 1.0e-7 end
                        )
                      )
                      |> Broca.NN.mult(lr)
                    )
                  )
                end
              )
            end
          )
        end)

      {updated_model, %Broca.Optimizers.Adam{adam | mv: new_mv, iter: iter}}
    end

    def batch_update(adam, model1, model2, learning_rate, cnt) do
      iter = if cnt == 0, do: adam.iter + 1, else: adam.iter

      lr =
        learning_rate * :math.sqrt(1.0 - :math.pow(adam.beta2, iter)) /
          (1.0 - :math.pow(adam.beta1, iter))

      new_mv = Broca.Optimizers.Adam.update_mv(adam, model2)

      updated_model =
        Enum.zip(new_mv, Enum.zip(model1, model2))
        |> Enum.map(fn {mv, {layer1, layer2}} ->
          Layer.batch_update(
            layer1,
            layer2,
            fn params, _ ->
              Keyword.keys(params)
              |> Enum.reduce(
                [],
                fn key, keyword ->
                  Keyword.put_new(
                    keyword,
                    key,
                    params[key]
                    |> Broca.NN.subtract(
                      Broca.NN.division(
                        elem(mv[key], 0),
                        Broca.NN.for_each(
                          elem(mv[key], 1),
                          fn val -> :math.sqrt(val) + 1.0e-7 end
                        )
                      )
                      |> Broca.NN.mult(lr)
                    )
                  )
                end
              )
            end
          )
        end)

      {updated_model, %Broca.Optimizers.Adam{adam | mv: new_mv, iter: iter}}
    end
  end

  def update_mv(adam, model) do
    Enum.zip(adam.mv, model)
    |> Enum.map(fn {mv, layer} ->
      if Map.has_key?(layer, :grads) do
        Keyword.keys(layer.grads)
        |> Enum.reduce(
          [],
          fn key, keyword ->
            Keyword.put_new(
              keyword,
              key,
              {
                layer.grads[key]
                |> Broca.NN.subtract(elem(mv[key], 0))
                |> Broca.NN.mult(1.0 - adam.beta1)
                |> Broca.NN.add(elem(mv[key], 0)),
                layer.grads[key]
                |> Broca.NN.for_each(fn val -> :math.pow(val, 2) end)
                |> Broca.NN.subtract(elem(mv[key], 1))
                |> Broca.NN.mult(1.0 - adam.beta2)
                |> Broca.NN.add(elem(mv[key], 1))
              }
            )
          end
        )
      else
        []
      end
    end)
  end
end
