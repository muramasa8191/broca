defprotocol Loss do
  def forward(layer, y, t)
  def backward(layer, t)
end

defmodule Broca.Losses.CrossEntropyError do
  defstruct val: nil

  @moduledoc """
  Cross Entropy Error
  """
  defimpl Loss, for: Broca.Losses.CrossEntropyError do
    @doc """
    Forward

    ## Examples
        iex> Loss.forward(%Broca.Losses.CrossEntropyError{}, [0.1, 0.4, 0.5], [0, 0, 1])
        0.6931469805599654
    """
    def forward(_, y, t) do
      Broca.NN.cross_entropy_error(y, t)
    end

    @doc """
    Backward

    ## Examples
      iex> Loss.backward(%Broca.Losses.CrossEntropyError{}, [0, 0, 1])
      [0, 0, 1]
    """
    def backward(_, t) do
      t
    end
  end
end
