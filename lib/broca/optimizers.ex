defmodule Broca.Optimizers do
  alias Broca.{
    Nif.NN,
    Optimizers
  }

  def init(:adam, params, learning_rate) do
    [
      :adam,
      Keyword.keys(params)
      |> Enum.reduce(
        Keyword.new(),
        fn key, keyword ->
          Keyword.put_new(keyword, key, {NN.zeros_like(params[key]), NN.zeros_like(params[key])})
        end
      )
      |> Keyword.put_new(:beta1, 0.9)
      |> Keyword.put_new(:beta2, 0.999)
      |> Keyword.put_new(:learning_rate, learning_rate)
    ]
  end

  def init(:sgd, _, learning_rate) do
    [:sgd, [learning_rate: learning_rate]]
  end

  def update_adam_mv(grads, config) do
    Enum.map(
      config,
      fn {key, val} ->
        case key do
          :beta1 ->
            {:beta1, val}

          :beta2 ->
            {:beta2, val}

          :learning_rate ->
            {:learning_rate, val}

          :iter ->
            {:iter, val}

          _ ->
            {m, v} = val

            {
              key,
              {
                grads[key]
                |> NN.subtract(m)
                |> NN.mult(1.0 - config[:beta1])
                |> NN.add(m),
                grads[key]
                |> NN.for_each(fn val -> :math.pow(val, 2) end)
                |> NN.subtract(v)
                |> NN.mult(1.0 - config[:beta2])
                |> NN.add(v)
              }
            }
        end
      end
    )
  end

  def adam_updata_params(learning_rate, params, config) do
    params
    |> Enum.map(fn {key, param} ->
      {
        key,
        param
        |> NN.subtract(
          NN.division(
            elem(config[key], 0),
            NN.for_each(elem(config[key], 1), fn val -> :math.sqrt(val) + 1.0e-7 end)
          )
          |> NN.mult(learning_rate)
        )
      }
    end)
  end

  def optimize(type, params, grads, config, first \\ true)

  def optimize(:none, _, _, _, _) do
    {nil, nil}
  end

  def optimize(:adam, params, grads, config, first) do
    config = if first, do: Keyword.update(config, :iter, 1, fn i -> i + 1 end), else: config

    lr =
      config[:learning_rate] * :math.sqrt(1.0 - :math.pow(config[:beta2], config[:iter])) /
        (1.0 - :math.pow(config[:beta1], config[:iter]))

    task = Task.async(Optimizers, :update_adam_mv, [grads, config])
    config = Task.await(task, :infinity)

    task = Task.async(Optimizers, :adam_updata_params, [lr, params, config])
    updated_params = Task.await(task, :infinity)

    {updated_params, config}
  end

  def optimize(:sgd, params, grads, config, _) do
    updated_params =
      Keyword.keys(params)
      |> Enum.reduce(
        Keyword.new(),
        fn key, keyword ->
          Keyword.put_new(
            keyword,
            key,
            sgd_optimize(params[:key], grads[key], config[:learning_rate])
          )
        end
      )

    {updated_params, config}
  end

  defp sgd_optimize(param, grad, learning_rate) when is_list(param) do
    Enum.zip(param, grad)
    |> Enum.map(fn {p, g} -> sgd_optimize(p, g, learning_rate) end)
  end

  defp sgd_optimize(param, grad, learning_rate) do
    param - grad * learning_rate
  end
end
