defmodule Broca.NN do
  @moduledoc """
  Module to handle `list` and `list of lists` for Neural Network.
  """

  @doc """
  Add `list2` to `list1` 

  ## Examples
    iex> Broca.NN.add([1, 2, 3], [4, 5, 6])
    [5, 7, 9]

    iex> Broca.NN.add([[1, 2], [3, 4]], [5, 6])
    [[6, 8], [8, 10]]

    iex> Broca.NN.add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[6, 8], [10, 12]]
  """
  @spec add([[number]], [[number]]) :: [[number]]
  @spec add([number], [number]) :: [number]
  def add(list1, list2) when is_list(hd list1) and is_list(hd list2) do
    Enum.zip(list1, list2)
    |> Enum.map(&add(elem(&1, 0), elem(&1, 1)))
  end
  def add(list1, list2) when is_list(hd list1) do
    list1
    |> Enum.map(&(add(&1, list2)))
  end
  def add(list1, list2) do
    Enum.zip(list1, list2)
    |> Enum.map(&(elem(&1, 0) + elem(&1, 1)))
  end

  @doc """
  Subtract `arg2` from `arg1`

  ## Examples
    iex> Broca.NN.subtract(10, 7)
    3

    iex> Broca.NN.subtract([1, 2, 3], 1)
    [0, 1, 2]

    iex> Broca.NN.subtract(1, [1, 2, 3])
    [0, -1, -2]

    iex> Broca.NN.subtract([3, 4, 5], [1, 2, 3])
    [2, 2, 2]

    iex> Broca.NN.subtract([[1, 2, 3], [11, 12, 13]], [1, 2, 1])
    [[0, 0, 2], [10, 10, 12]]

    iex> Broca.NN.subtract([[99, 98], [97, 96]], [[1, 2], [3, 4]])
    [[98, 96], [94, 92]]
  """
  def subtract(list1, list2) when is_list(hd list1) and is_list(hd list2) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {xs, ys} -> subtract(xs, ys) end)
  end
  def subtract(list1, list2) when is_list(hd list1) and is_list(list2) do
    list1
    |> Enum.map(&(subtract(&1, list2)))
  end
  def subtract(list1, list2) when is_list(list1) and is_list(list2) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {x, y} -> subtract(x, y) end)
  end
  def subtract(list, y) when is_list(list) do
    list
    |> Enum.map(&(subtract(&1, y)))
  end
  def subtract(x, list) when is_list(list) do
    list
    |> Enum.map(&(subtract(x, &1)))
  end
  def subtract(x, y) do
    x - y
  end

  @doc """
  Vector multiplication

  ## Examples
    iex> Broca.NN.mult([1, 2, 3], [4, 5, 6])
    [4, 10, 18]

    iex> Broca.NN.mult([[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]])
    [[4, 10, 18], [4, 10, 18]]

    iex> Broca.NN.mult(10, 20)
    200
  """
  def mult(list1, list2) when is_list(hd list1) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {xs, ys} -> mult(xs, ys) end)
  end
  def mult(list1, list2) when is_list list1 do
    Enum.zip(list1, list2)
    |> Enum.map(fn {x, y} -> x * y end)
  end
  def mult(x, y) do
    x * y
  end

  @doc """
  Transpose the `list` given

  ## Examples
    iex> Broca.NN.transpose([1, 2, 3])
    [[1], [2], [3]]

    iex> Broca.NN.transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
  """
  @spec transpose([number]) :: [number]
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

  @doc """
  Dot product

  ## Examples
    iex> a = [1, 2, 3]
    iex> b = [4, 5, 6]
    iex> Broca.NN.dot(a, b)
    32

    iex> a = [[1, 2], [3, 4], [5, 6]]
    iex> b = [7, 8]
    iex> Broca.NN.dot(a, b)
    [23, 53, 83]

    iex> a = [[1, 2], [3, 4]]
    iex> b = [[5, 6], [7, 8]]
    iex> Broca.NN.dot(a, b)
    [[19, 22], [43, 50]]

    iex> a = [1, 2]
    iex> b = [[1, 3, 5], [2, 4, 6]]
    iex> Broca.NN.dot(a, b)
    [5, 11, 17]
  """
  @spec dot([[number]], [[number]]) :: [[number]]
  @spec dot([[number]], [number]) :: [number]
  @spec dot([number], [[number]]) :: [number]
  @spec dot([number], [number]) :: number
  def dot(as, bs) when is_list(hd as) and is_list(hd bs) do
    bt = transpose(bs)
    as
    |> Enum.map(fn a ->Enum.map(bt, fn b -> dot(a, b) end) end)
  end
  def dot(as, b) when is_list(hd as) do
    as
    |> Enum.map(&(dot(&1, b)))
  end
  def dot(a, bs) when is_list(hd bs) do
    transpose(bs)
    |> Enum.map(&(dot(a, &1)))
  end
  def dot(a, b) do
    Enum.zip(a, b)
    |> Enum.reduce(0, &(&2 + elem(&1, 0) * elem(&1, 1)))
  end

  @doc """
  Sum the `list`

  ## Examples
    iex> Broca.NN.sum([1, 2, 3])
    6

    iex> Broca.NN.sum([[1, 2, 3], [4, 5, 6]], :row)
    [6, 15]

    iex> Broca.NN.sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]], :col)
    [12, 15, 18]
  """
  def sum(list) do
    sum(list, :row)
  end
  def sum(list, axis) when axis == :col do
    arr = List.duplicate(0, length(hd list))
    list
    |> Enum.reduce(arr, fn sub_list, arr ->
      Enum.zip(sub_list, arr)
      |> Enum.map(fn {x, acc} -> x + acc end)
    end)
  end
  def sum(list, _) when is_list(hd list) do
    list |> Enum.map(&(Enum.sum(&1)))
  end
  def sum(list, _) do
    list |> Enum.sum
  end

  @doc """
  Sigmoid function

  ## Examples
    iex> Broca.NN.sigmoid([-1.0, 1.0, 2.0])
    [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]

    iex> Broca.NN.sigmoid([[-1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]])
    [[0.2689414213699951, 0.7310585786300049, 0.8807970779778823],
     [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]]
  """
  @spec sigmoid([[number]]) :: [[number]]
  @spec sigmoid([number]) :: [number]
  def sigmoid(list) when is_list(hd list) do
    list |> Enum.map(&(sigmoid(&1)))
  end
  def sigmoid(list) do
    list |> Enum.map(&(1 / (1 + :math.exp(-&1))))
  end

  @doc """
  ReLU function

  ## Examples
    iex> Broca.NN.relu([-1.0, 0.0, 1.0, 2.0])
    [0.0, 0.0, 1.0, 2.0]

    iex> Broca.NN.relu([[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0]])
    [[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 1.0, 2.0]]
  """
  @spec relu([[number]]) :: [[number]]
  @spec relu([number]) :: [number]
  def relu(list) when is_list(hd list) do
    list |> Enum.map(&(relu(&1)))
  end
  def relu(list) do
    list |> Enum.map(&(max(0.0, &1)))
  end

  @doc """
  Softmax function

  ## Examples
    iex> Broca.NN.softmax([0.3, 2.9, 4.0])
    [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]

    iex> Broca.NN.softmax([[0.3, 2.9, 4.0], [0.3, 2.9, 4.0], [0.3, 2.9, 4.0]])
    [[0.01821127329554753, 0.24519181293507392, 0.7365969137693786],
     [0.01821127329554753, 0.24519181293507392, 0.7365969137693786],
     [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]]
  """
  @spec softmax([[number]]) :: [[number]]
  @spec softmax([number]) :: [number]
  def softmax(list) when is_list(hd list) do
    list |> Enum.map(&(softmax(&1)))
  end
  def softmax(list) do
    max = Enum.max(list)
    {sum, res} = list 
                 |> Enum.reverse
                 |> Enum.reduce({0, []}, 
                    fn x, {s, r} -> 
                      ex = :math.exp(x - max)
                      {s + ex, [ex]++r}
                    end
                  )
    res |> Enum.map(&(&1 / sum))
  end

  @doc """
  Convert class list to one hot vector.

  ## Examples
    iex> Broca.NN.one_hot(0, 4)
    [1, 0, 0, 0, 0]

    iex> Broca.NN.one_hot(3, 4)
    [0, 0, 0, 1, 0]

    iex> Broca.NN.one_hot([0, 1, 2, 3, 4], 4)
    [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]]
  """
  def one_hot(list, max) when is_list list do
    list |> Enum.map(&(one_hot(&1, max)))
  end
  def one_hot(class, max) do
    0..max |> Enum.map(&(if &1 == class, do: 1, else: 0))
  end

  @doc """
  Calculate cross entropy error.

  ## Examples
    iex> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    iex> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    iex> Broca.NN.cross_entropy_error(y, t)
    0.51082545709933802
    
    iex> t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    iex> y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
    iex> Broca.NN.cross_entropy_error(y, t)
    1.021650914198676
  """
  def cross_entropy_error(ys, ts) when is_list(hd ys) do
    Enum.zip(ys, ts)
    |> Enum.reduce(0, fn {y, t}, acc -> acc + cross_entropy_error(y, t) end)
  end
  def cross_entropy_error(ys, ts) do
      delta = 1.0e-7
      Enum.zip(ys, ts)
      |> Enum.reduce(0, fn {y, t}, acc -> if t == 0, do: acc, else: acc - :math.log(y + delta) end)
  end

  @doc """
  Get the index of maximum value in the list

  ## Examples
    iex> Broca.NN.argmax([5.0, 1.0, 2.0, 3.0])
    0

    iex> Broca.NN.argmax([1.0, 4.0, 2.0, 3.0])
    1

    iex> Broca.NN.argmax([[5.0, 1.0, 2.0, 3.0], [1.0, 4.0, 2.0, 3.0]])
    [0, 1]
  """
  def argmax(list) when is_list(hd list) do
    list |> Enum.map(&(argmax(&1)))
  end
  def argmax(list) do
    {res, _, _} =
      list |> Enum.reduce({-1, -999999, 0}, fn x, {idx, max, cur} -> if x > max, do: {cur, x, cur+1}, else: {idx, max, cur+1} end)
    res
  end

  @doc """
  Mask the list with the filter given

  ## Examples
    iex> Broca.NN.filter_mask([-1.0, 0.2, 1.0, -0.3], fn x -> x <= 0 end)
    [True, False, False, True]

    iex> Broca.NN.filter_mask([[-1.0, 0.2, 1.0, -0.3], [-1.0, 0.2, 1.0, -0.3]], fn x -> x <= 0 end)
    [[True, False, False, True], [True, False, False, True]]
  """
  def filter_mask(list, filter) when is_list(hd list) do
    list
    |> Enum.map(&(filter_mask(&1, filter)))
  end
  def filter_mask(list, filter) do
    list
    |> Enum.map(&(if filter.(&1), do: True, else: False))
  end
end