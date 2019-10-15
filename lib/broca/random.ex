defmodule Broca.Random do
  @doc """
  Generate the list with `n` random elements.

  ## Examples
      iex> length(Broca.Random.randn(200))
      200
  """
  def randn(n) do
    1..n |> Enum.map(fn _ -> :rand.normal() end)
  end

  @doc """
  Generate the list of lists with `n` x `m` random elements.

  ## Examples
      iex> r = Broca.Random.randn(3, 5)
      iex> {length(r), length(hd r)}
      {3, 5}
  """
  def randn(n, m) do
    1..n |> Enum.map(fn _ -> 1..m |> Enum.map(fn _ -> :rand.normal() end) end)
  end

  @doc """
  Generate 3D random list with `n` x `m` x `l` dims.

  ## Examples
      iex> r = Broca.Random.randn(2, 3, 4)
      iex> {length(r), length(hd r), length(hd (hd r))}
      {2, 3, 4}
  """
  def randn(n, m, l) do
    for _ <- 1..n do
      for _ <- 1..m do
        for _ <- 1..l do
          :rand.normal()
        end
      end
    end
  end

  @doc """
  Generate 4D random list with `n` x `m` x `l` x `k` dims.

  ## Examples
      iex> r = Broca.Random.randn(3, 1, 28, 28)
      iex> {length(r), length(hd r), length(hd (hd r)), length(hd (hd (hd r)))}
      {3, 1, 28, 28}
  """
  def randn(n, m, l, k) do
    for _ <- 1..n do
      for _ <- 1..m do
        for _ <- 1..l do
          for _ <- 1..k do
            :rand.normal()
          end
        end
      end
    end
  end
end
