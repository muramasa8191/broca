defmodule Broca.Optimizers.AdaGrad do
  defstruct h: nil, init: False

  defimpl Broca.Optimizer, for: Broca.Optimizers.AdaGrad do
    @doc """
    Constructor.

    ## Examples
      iex> Broca.Optimizer.init(%Broca.Optimizers.AdaGrad{}, [%Broca.Layers.Affine{params: [bias: [5, 6], weight: [[1, 2], [3, 4]]]}, \
          %Broca.Activations.ReLU{}, %Broca.Layers.Affine{params: [bias: [8, 9, 10], weight: [[1, 2, 3], [4, 5, 6]]]}, \
          %Broca.Activations.Softmax{}])
      %Broca.Optimizers.AdaGrad{h: [[weight: [[0.0, 0.0], [0.0, 0.0]], bias: [0.0, 0.0]], [], [weight: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], bias: [0.0, 0.0, 0.0]], []], init: true}
    """
    def init(_, model) do
      h =
        model
        |> Enum.map(fn layer ->
          if Map.has_key?(layer, :params) do
            Keyword.keys(layer.params)
            |> Enum.reduce(
              Keyword.new(),
              &Keyword.put_new(&2, &1, Broca.NN.zeros_like(layer.params[&1]))
            )
          else
            []
          end
        end)

      %Broca.Optimizers.AdaGrad{h: h, init: true}
    end

    def update(optimizer, model, learning_rate) do
      h =
        Enum.zip(optimizer.h, model)
        |> Enum.map(fn {opt, layer} ->
          if Map.has_key?(layer, :grads) do
            Keyword.keys(layer.grads)
            |> Enum.reduce(
              [],
              fn key, keyword ->
                Keyword.put_new(
                  keyword,
                  key,
                  Broca.Optimizers.AdaGrad.update_h(opt, Keyword.get(layer.grads, key))
                )
              end
            )
          else
            []
          end
        end)

      updated_model =
        Enum.zip(h, model)
        |> Enum.map(fn {h2, layer} ->
          Layer.update(layer, fn param, grad ->
            Broca.Optimizers.AdaGrad.optimize(param, grad, h2, learning_rate)
          end)
        end)

      {updated_model, %Broca.Optimizers.AdaGrad{init: true, h: h}}
    end

    def batch_update(optimizer, model1, model2, learning_rate, _) do
      h =
        Enum.zip(optimizer.h, model2)
        |> Enum.map(fn {opt, layer} ->
          if Enum.empty?(opt) do
            []
          else
            Keyword.keys(layer.grads)
            |> Enum.reduce(
              [],
              &Keyword.put_new(
                &2,
                &1,
                Broca.Optimizers.AdaGrad.update_h(
                  Keyword.get(opt, &1),
                  Keyword.get(layer.grads, &1)
                )
              )
            )
          end
        end)

      updated_model =
        Enum.zip(model1, model2)
        |> Enum.zip(h)
        |> Enum.map(fn {{layers1, layers2}, h2} ->
          Layer.batch_update(layers1, layers2, fn params, grads ->
            Keyword.keys(grads)
            |> Enum.reduce(
              [],
              fn key, keyword ->
                Keyword.put_new(
                  keyword,
                  key,
                  Broca.Optimizers.AdaGrad.optimize(
                    Keyword.get(params, key),
                    Keyword.get(grads, key),
                    Keyword.get(h2, key),
                    learning_rate
                  )
                )
              end
            )
          end)
        end)

      {updated_model, %Broca.Optimizers.AdaGrad{init: true, h: h}}
    end
  end

  def update_h(_, []) do
    []
  end

  def update_h(hs, grads) when is_list(hs) do
    Enum.zip(hs, grads)
    |> Enum.map(fn {h, g} -> update_h(h, g) end)
  end

  def update_h(h, grad) do
    h + grad * grad
  end

  def optimize(param, grad, h, learning_rate) when length(h) > 0 do
    Enum.zip(param, grad)
    |> Enum.zip(h)
    |> Enum.map(fn {{p, g}, h2} ->
      optimize_param(p, g, h2, learning_rate)
    end)
  end

  def optimize_param(params, grads, hs, learning_rate) when is_list(params) do
    Enum.zip(params, grads)
    |> Enum.zip(hs)
    |> Enum.map(fn {{param, grad}, h} -> optimize_param(param, grad, h, learning_rate) end)
  end

  def optimize_param(param, grad, h, learning_rate) do
    param - learning_rate * grad / (:math.sqrt(h) + 1.0e-7)
  end
end
