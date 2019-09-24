defprotocol Optimizer do
  def init(optimizer, model)
  def update(optimizer, model, learning_rate)
  def batch_update(optimizers)
end

defmodule Broca.Optimizers.AdaGrad do
  defstruct h: nil, init: False

  defimpl Optimizer, for: Broca.Optimizers.AdaGrad do
    @doc """
    Constructor.

    ## Examples
      iex> Optimizer.init(%Broca.Optimizers.AdaGrad{}, [%Broca.Layers.Affine{grads: [weight: [[1, 2], [3, 4]], bias: [5, 6]]}, \
          %Broca.Activations.ReLU{}, %Broca.Layers.Affine{grads: [weight: [[1, 2, 3], [4, 5, 6]], bias: [8, 9, 10]]}, \
          %Broca.Activations.Softmax{}])
      %Broca.Optimizers.AdaGrad{h: [[[[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0]], [], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0.0, 0.0, 0.0]], []], init: True}
    """
    def init(_, model) do
      h = model |> Enum.map(&(Layer.get_grads(&1) |> Broca.NN.zeros_like))
      %Broca.Optimizers.AdaGrad{h: h, init: True}
    end
    def update(optimizer, model, learning_rate) do
      h = Enum.zip(optimizer.h, model)
        |> Enum.map(fn {opt, mod} -> Broca.Optimizers.AdaGrad.update_h(opt, Layer.get_grads(mod)) end)

      updated_model =
        Enum.zip(h, model)
        |> Enum.map(fn {h2, layer} -> Layer.update(layer, fn (param, grad, idx) -> Broca.Optimizers.AdaGrad.optimize(param, grad, h2, learning_rate, idx) end) end)

      {updated_model, %Broca.Optimizers.AdaGrad{init: True, h: h}}
    end

    def batch_update(optimizers) do
      hd optimizers
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
    h + (grad * grad)
  end

  def optimize(param, grad, h, learning_rate, idx) when idx == -1 do  
    optimize(param, grad, h, learning_rate)
  end
  def optimize(param, grad, h, learning_rate, idx) when idx == 0 do  
    optimize(param, grad, (hd h), learning_rate, idx-1)
  end
  def optimize(param, grad, h, learning_rate, idx) do  
    optimize(param, grad, (tl h), learning_rate, idx-1)
  end

  def optimize(param, grad, h, learning_rate) when is_list(grad) do  
    # if length(param) != length(h) do
    #   raise("size not match: param(#{length(param)}) != h(#{length(h)})")
    # else
    #   IO.puts("size match: param(#{length(param)}) != h(#{length(h)})")
    # end
    Enum.zip(param, grad)
    |> Enum.zip(h)
    |> Enum.map(fn {{p, g}, h2} -> optimize(p, g, h2, learning_rate) end)
  end
  def optimize(param, grad, h, learning_rate) do
    # if is_list(h) or is_list(param) or is_list(grad), do: raise("list h:#{IO.inspect(h)}, param: #{IO.inspect(param)}, grad: #{IO.inspect(grad)}")
    param - (learning_rate * grad / (:math.sqrt(h) + 1.0e-7))
  end
end

defmodule Broca.Optimizers.SGD do
  defstruct init: True
  defimpl Optimizer, for: Broca.Optimizers.SGD do
    def init(_, _) do
      %Broca.Optimizers.SGD{}
    end

    def update(_, model, learning_rate) do
      updated_model =
        model
        |> Enum.map(&(Layer.update(&1, fn (param, grad) -> Broca.Optimizers.SGD.update_param(param, grad, learning_rate) end)))

      # [_, _, layer, _] = updated_model
      # IO.puts("updated")
      # IO.puts("layer.params")
      # IO.inspect(layer.params)
      # IO.puts("layer.grads")
      # IO.inspect(layer.grads)

      {updated_model, %Broca.Optimizers.SGD{}}
    end

    def batch_update(optimizers) do
      hd optimizers
    end
  end


  def update_param(params, grads, learning_rate) when is_list(params) do
    Enum.zip(params, grads)
    |> Enum.map(fn {param, grad} -> update_param(param, grad, learning_rate) end)
  end
  def update_param(param, grad, learning_rate) do
    param - (grad * learning_rate)
  end

end