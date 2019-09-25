defmodule Broca.NumericalGradient do
  @h 1.0e-4

  @doc """
  Numerical Gradient

  ## Examples
      iex> Broca.NumericalGradient.numerical_gradient(fn idx, diff -> Broca.NumericalGradient.dot2([1, 2, 3], [4, 5, 6], idx, diff) end, [weight: [4, 5, 6]], :weight)
      [1.000000000015433, 1.9999999999953388, 2.9999999999752447]
  """
  def numerical_gradient(func, params, key) do
    list = params[key]

    if is_list(hd(list)) do
      0..(length(list) - 1)
      |> Enum.map(fn idx1 ->
        0..(length(hd(list)) - 1)
        |> Enum.map(fn idx2 ->
          (
            # IO.puts("#{idx1 + 1}/#{length(list)}, #{idx2 + 1}/#{length(hd(list))}")
            func.(idx1, idx2, @h) - func.(idx1, idx2, -@h)
          ) / (2 * @h)
        end)
      end)
    else
      0..(length(list) - 1)
      |> Enum.map(fn idx -> (func.(idx, @h) - func.(idx, -@h)) / (2 * @h) end)
    end
  end

  @doc """
  Dot product for numerical gradient

  ## Examples
      iex> Broca.NumericalGradient.dot2([1, 2, 3], [4, 5, 6], 1, 1.0e-4)
      32.0002

      iex> Broca.NumericalGradient.dot2([1, 2], [[1, 2, 3,], [4, 5, 6]], 1, 1, 1.0e-4)
      [9, 12.0002, 15]

      iex> Broca.NumericalGradient.dot2([[1, 2], [3, 4]], [[1, 2, 3,], [4, 5, 6]], 1, 1, 1.0e-4)
      [[9, 12.0002, 15], [19, 26.0004, 33]]
  """
  def dot2(as, bs, idx1, idx2, diff) do
    bt = bs |> Broca.NN.transpose() |> Enum.with_index()

    if is_list(hd(as)) do
      as
      |> Enum.map(fn a ->
        Enum.map(bt, fn {b, idx} ->
          dot2(a, b, if(idx1 == idx, do: idx2, else: -1), diff)
        end)
      end)
    else
      bt |> Enum.map(fn {b, idx} -> dot2(as, b, if(idx1 == idx, do: idx2, else: -1), diff) end)
    end
  end

  def dot2(a, b, target_idx, diff) do
    if is_list(hd(a)) do
      a
      |> Enum.map(fn a2 ->
        Enum.zip(a2, Enum.with_index(b))
        |> Enum.reduce(0, fn {a3, {b, idx}}, acc ->
          if idx == target_idx, do: acc + a3 * (b + diff), else: acc + a3 * b
        end)
      end)
    else
      Enum.zip(a, Enum.with_index(b))
      |> Enum.reduce(0, fn {a, {b, idx}}, acc ->
        if idx == target_idx, do: acc + a * (b + diff), else: acc + a * b
      end)
    end
  end

  @doc """
  Add for numerical gradient

  ## Examples
      iex> Broca.NumericalGradient.add2([1, 2, 3], [4, 5, 6], 2, 1.0e-4)
      [5, 7, 9.0001]
  """
  def add2(as, bs, idx1, idx2, diff) do
    bt = bs |> Broca.NN.transpose() |> Enum.with_index()

    if is_list(hd(as)) do
      as
      |> Enum.map(fn a ->
        Enum.map(bt, fn {b, idx} ->
          add2(a, b, if(idx1 == idx, do: idx2, else: -1), diff)
        end)
      end)
    else
      bt |> Enum.map(fn {b, idx} -> add2(as, b, if(idx1 == idx, do: idx2, else: -1), diff) end)
    end
  end

  def add2(a, b, target_idx, diff) do
    if is_list(hd(a)) do
      a
      |> Enum.map(fn a2 ->
        Enum.zip(a2, Enum.with_index(b))
        |> Enum.map(fn {a3, {b, idx}} ->
          if idx == target_idx, do: a3 + b + diff, else: a3 + b
        end)
      end)
    else
      Enum.zip(a, Enum.with_index(b))
      |> Enum.map(fn {a, {b, idx}} ->
        if idx == target_idx, do: a + (b + diff), else: a + b
      end)
    end
  end
end
