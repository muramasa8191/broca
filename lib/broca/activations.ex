defmodule Broca.Activations.ReLU do
  defstruct mask: nil

  defimpl Layer, for: Broca.Activations.ReLU do
    @doc """
    Forward

    ## Examples
        iex> Layer.forward(%Broca.Activations.ReLU{}, [-1.0, 2.0, 3.0, -4.0])
        {%Broca.Activations.ReLU{mask: [True, False, False, True]}, [0.0, 2.0, 3.0, 0.0]}
    """
    def forward(_, x) do
      mask = x |> Broca.NN.filter_mask(fn val -> val <= 0.0 end)
      {%Broca.Activations.ReLU{mask: mask}, Broca.NN.relu(x)}
    end

    @doc """
    Backward

    ## Examples
        iex> Layer.backward(%Broca.Activations.ReLU{mask: [True, False, False, True]}, [100.0, 20.0, 30.0, 24.0])
        {%Broca.Activations.ReLU{mask: [True, False, False, True]}, [0.0, 20.0, 30.0, 0.0]}
    """
    def backward(layer, dout) do
      res =
        Enum.zip(layer.mask, dout)
        |> Enum.map(fn {mask, dout} -> if mask == True, do: 0.0, else: dout end)

      {layer, res}
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
      res = dout |> Broca.NN.mult(Broca.NN.subtract(1.0, layer.out)) |> Broca.NN.mult(layer.out)
      {layer, res}
    end
  end
end
