defmodule Broca.NN do
  @moduledoc """
  Module to handle list and list of lists.
  """

  @doc """
  Transpose the `list` given

  ## Examples
    iex> Broca.NN.transpose([1, 2, 3])
    [[1], [2], [3]]

    iex> Broca.NN.transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
  """
  def transpose(list) when is_list(hd list) do
    arr = List.duplicate([], length(hd list))
    list 
    |> Enum.reverse
    |> Enum.reduce(arr, fn (sub_list, arr) ->
      Enum.zip(sub_list, arr) 
      |> Enum.map(&([elem(&1, 0)]++elem(&1, 1))) 
    end)
  end
  def transpose(list) do
    list |> Enum.map(&([&1]))
  end

  def dot(aa, bb) when is_list(hd aa) do
    bbt = transpose(bb)
    aa
    |> Enum.with_index
    |> Flow.from_enumerable(max_demand: 1)
    |> Flow.map(fn {list, idx} -> {_dot_row(list, bbt), idx} end)
    |> Enum.sort_by(&elem(&1, 1))
    |> Enum.map(&elem(&1, 0))
  end
  def dot(a, b) do
    _dot_calc(a, b)
  end
  defp _dot_row(a, b) do
    b |> Enum.map(&(_dot_calc(a, &1)))
  end
  defp _dot_calc(a, b) do
    Enum.zip(a, b)
    |> Enum.reduce(0, fn ({a, b}, acc) -> Float.floor(acc + a * b, 8) end)
  end

  @doc """
  Sigmoid function

  ## Examples
    iex> Broca.NN.sigmoid([-1.0, 1.0, 2.0])
    [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]
  """
  def sigmoid(list) do
    list |> Enum.map(&(1 / (1 + :math.exp(-&1))))
  end

  @doc """
  ReLU function

  ## Examples
    iex> Broca.NN.relu([-1.0, 0.0, 1.0, 2.0])
    [0.0, 0.0, 1.0, 2.0]
  """
  def relu(list) do
    list |> Enum.map(&(max(0.0, &1)))
  end


  # def softmax(list) do
  #   sum = list |> List.foldr(0, &(&1 + &2))
  # end
end