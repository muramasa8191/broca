defmodule Broca.NumericalGradient do
  @h 1.0e-4
  def numerical_gradient(func, params, key) do
    list = params[key]
    if (is_list(hd list)) do
      0..length(list)-1
      |> Enum.map(
        fn idx1 -> 0..length(hd list)-1
                   |> Enum.map(&(func.(idx1, &1, list)))
        end)
    else
      0..length(list)-1
      |> Enum.map(&(func.(&1, list)))
    end
  end

  def dot2(as, bs, idx1, idx2) when is_number(idx2) do
    bt = bs |> Enum.with_index |> Broca.NN.transpose
    if is_list(hd as) do
      as
      |> Enum.map(fn a ->
                    Enum.map(bt, fn {b, idx} ->
                      dot2(a, b, idx2, idx1 == idx)
                    end)
                  end) 
    else
      bt |> Enum.map(fn {b, idx} -> dot2(as, b, idx2, (if idx1 == idx, do: idx, else: -1))end)
    end
  end

  @doc """
  Dot product for numerical gradient

  ## Examples
      iex> Broca.NumericalGradient.dot2([1, 2, 3], [4, 5, 6], 1)
      32.0002
  """
  def dot2(a, b, target_idx) do
    Enum.zip(a, Enum.with_index(b))
    |> Enum.reduce(0, fn {a, {b, idx}}, acc ->
      if idx == target_idx, do: acc + (a * (b + @h)), else: acc + (a * b)
    end)
  end


  def dot(a, b) do
    Enum.zip(a, b)
    |> Enum.reduce(0, &(&2 + elem(&1, 0) * elem(&1, 1)))
  end

end