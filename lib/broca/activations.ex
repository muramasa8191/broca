defmodule Broca.Activations.ReLU do
  defstruct mask: nil

  def to_string(relu) do
    "ReLU: mask=#{Broca.NN.shape_string(relu.mask)}"
  end

  defimpl Inspect, for: Broca.Activations.ReLU do
    def inspect(relu, _) do
      Broca.Activations.ReLU.to_string(relu)
    end
  end

  defimpl String.Chars, for: Broca.Activations.ReLU do
    def to_string(relu) do
      Broca.Activations.ReLU.to_string(relu)
    end
  end

  defimpl Layer, for: Broca.Activations.ReLU do
    @doc """
    Forward

    ## Examples
        iex> Layer.forward(%Broca.Activations.ReLU{}, [-1.0, 2.0, 3.0, -4.0])
        {%Broca.Activations.ReLU{mask: [True, False, False, True]}, [0.0, 2.0, 3.0, 0.0]}
    """
    def forward(_, x) do
      mask = x |> Broca.NN.filter_mask(fn val -> val <= 0.0 end)
      out = Enum.zip(mask, x) |> Enum.map(fn {m, v} -> Broca.NN.mask(m, v, 0.0) end)
      {%Broca.Activations.ReLU{mask: mask}, out}
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Activations.ReLU{mask: [True, False, False, True]}, [100.0, 20.0, 30.0, 24.0])
        {%Broca.Activations.ReLU{mask: nil}, [0.0, 20.0, 30.0, 0.0]}

        iex> Layer.backward(%Broca.Activations.ReLU{mask: [[True, False, False, True], [True, False, False, True]]}, [[100.0, 20.0, 30.0, 24.0], [100.0, 20.0, 30.0, 24.0]])
        {%Broca.Activations.ReLU{mask: nil}, [[0.0, 20.0, 30.0, 0.0], [0.0, 20.0, 30.0, 0.0]]}
    """
    def backward(layer, dout) do
      res =
        Enum.zip(layer.mask, dout)
        |> Enum.map(fn {m, v} -> Broca.NN.mask(m, v, 0.0) end)

      {%Broca.Activations.ReLU{}, res}
    end

    def update(layer, _) do
      layer
    end

    def batch_update(_, layer, _) do
      layer
    end
  end
end

defmodule Broca.Activations.Sigmoid do
  defstruct out: []

  defimpl Layer, for: Broca.Activations.Sigmoid do
    @doc """
    Forward

    ## Examples
        iex> Layer.forward(%Broca.Activations.Sigmoid{}, [-1.0, 0.0, 1.0, 2.0])
        {%Broca.Activations.Sigmoid{
          out: [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]},
        [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]}
    """
    def forward(_, x) do
      out = Broca.NN.sigmoid(x)
      {%Broca.Activations.Sigmoid{out: out}, out}
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Activations.Sigmoid{out: [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]}, [0.8, 1.2, -0.3, 2.0])
        {%Broca.Activations.Sigmoid{out: [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]},
         [0.15728954659318548, 0.3, -0.05898357997244455, 0.20998717080701323]}
    """
    def backward(layer, dout) do
      res =
        dout
        |> Broca.NN.mult(Broca.NN.subtract(1.0, layer.out))
        |> Broca.NN.mult(layer.out)

      {layer, res}
    end

    def update(layer, _) do
      layer
    end

    def batch_update(_, layer, _) do
      layer
    end
  end
end

defmodule Broca.Activations.Softmax do
  defstruct y: []

  def to_string(softmax) do
    "Softmax: y=#{Broca.NN.shape_string(softmax.y)}"
  end

  defimpl Inspect, for: Broca.Activations.Softmax do
    def inspect(softmax, _) do
      Broca.Activations.Softmax.to_string(softmax)
    end
  end

  defimpl String.Chars, for: Broca.Activations.Softmax do
    def to_string(softmax) do
      Broca.Activations.Softmax.to_string(softmax)
    end
  end

  defimpl Layer, for: Broca.Activations.Softmax do
    @doc """
    Forward for Softmax

    ## Examples
        iex> Layer.forward(%Broca.Activations.Softmax{}, [0.3, 2.9, 4.0])
        {%Broca.Activations.Softmax{y: [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]},
         [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]}
    """
    def forward(_, x) do
      out = Broca.NN.softmax(x)
      {%Broca.Activations.Softmax{y: out}, out}
    end

    @doc """
    Backward for Softmax

    ## Examples
        iex> Layer.backward(%Broca.Activations.Softmax{y: [[0.1, 0.5, 0.4], [0.2, 0.2, 0.6]]}, [[0, 1, 0],[0, 0, 1]])
        {%Broca.Activations.Softmax{}, [[0.05, -0.25, 0.2], [0.1, 0.1, -0.2]]}
    """
    def backward(layer, dout) do
      batch_size = length(dout)
      res = Broca.NN.subtract(layer.y, dout) |> Broca.NN.division(batch_size)

      {%Broca.Activations.Softmax{}, res}
    end

    def update(layer, _) do
      layer
    end

    def batch_update(_, layer, _) do
      layer
    end
  end
end

defmodule Broca.Activations do
  def create(activation_type) do
    case activation_type do
      :relu -> %Broca.Activations.ReLU{}
      :sigmoid -> %Broca.Activations.Sigmoid{}
      :softmax -> %Broca.Activations.Softmax{}
      _ -> nil
    end
  end
end
