defprotocol Broca.Optimizer do
  def init(optimizer, model)
  def update(optimizer, model, learning_rate)
  def batch_update(optimizer, model1, model2, learning_rate, cnt)
end

defmodule Broca.Optimizers.SGD do
  defstruct init: True

  defimpl Broca.Optimizer, for: Broca.Optimizers.SGD do
    def init(_, _) do
      %Broca.Optimizers.SGD{}
    end

    def update(_, model, learning_rate) do
      updated_model =
        model
        |> Enum.map(
          &Layer.update(&1, fn param, grad ->
            Broca.Optimizers.SGD.update_param(param, grad, learning_rate)
          end)
        )

      {updated_model, %Broca.Optimizers.SGD{}}
    end

    def batch_update(_, model1, model2, learning_rate, _) do
      updated_model =
        Enum.zip(model1, model2)
        |> Enum.map(
          &Layer.batch_update(elem(&1, 0), elem(&1, 1), fn params, grads ->
            Keyword.keys(grads)
            |> Enum.reduce([],
              fn key, keyword ->
                Keyword.put_new(
                  keyword,
                  key,
                  Broca.Optimizers.SGD.update_param(params[key], grads[key], learning_rate)
                )
              end)
          end)
        )

      {updated_model, %Broca.Optimizers.SGD{}}
    end
  end

  def update_params(params, grads, learning_rate) do
    Keyword.keys(params)
    |> Enum.reduce(
      [],
      &Keyword.put_new(
        &2,
        &1,
        update_param(Keyword.get(params, &1), Keyword.get(grads, &1), learning_rate)
      )
    )
  end

  def update_param(params, grads, learning_rate) when is_list(params) do
    Enum.zip(params, grads)
    |> Enum.map(fn {param, grad} -> update_param(param, grad, learning_rate) end)
  end

  def update_param(param, grad, learning_rate) do
    param - grad * learning_rate
  end
end

defmodule Broca.Optimizers do
  def create(type, model) do
    case type do
      :adam -> Broca.Optimizer.init(%Broca.Optimizers.Adam{}, model)
      :adaGrad -> Broca.Optimizer.init(%Broca.Optimizers.AdaGrad{}, model)
      _ -> Broca.Optimizer.init(%Broca.Optimizers.SGD{}, model)
    end
  end
end
