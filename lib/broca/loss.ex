defmodule Broca.Loss do
  alias Broca.Nif.NN

  def loss(:cross_entropy_error, y, t) do
    NN.cross_entropy_error(y, t)
  end

  def backward(:cross_entropy_error, _, t) do
    t
  end
end
