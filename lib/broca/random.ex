defmodule Broca.Random do
  @doc """
  Generate the list with `n` random elements.
  """
  def randn(n) do
    1..n |> Enum.map(fn _ -> :rand.normal end)
  end

  @doc """
  Generate the list of lists with `n` x `m` random elements.
  """
  def randn(n, m) do
    1..n |> Enum.map(fn _ -> 1..m |> Enum.map(fn _ -> :rand.normal end) end)
  end
end