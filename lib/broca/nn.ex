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
  def add(list1, list2) when is_list(hd(list1)) and is_list(hd(list2)) do
    Enum.zip(list1, list2)
    |> Enum.map(&add(elem(&1, 0), elem(&1, 1)))
  end

  def add(list1, list2) when is_list(hd(list1)) do
    list1
    |> Enum.map(&add(&1, list2))
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
  def subtract(list1, list2) when is_list(hd(list1)) and is_list(hd(list2)) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {xs, ys} -> subtract(xs, ys) end)
  end

  def subtract(list1, list2) when is_list(hd(list1)) and is_list(list2) do
    list1
    |> Enum.map(&subtract(&1, list2))
  end

  def subtract(list1, list2) when is_list(list1) and is_list(list2) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {x, y} -> subtract(x, y) end)
  end

  def subtract(list, y) when is_list(list) do
    list
    |> Enum.map(&subtract(&1, y))
  end

  def subtract(x, list) when is_list(list) do
    list
    |> Enum.map(&subtract(x, &1))
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
  def mult(list1, list2) when is_list(hd(list1)) and is_list(hd(list2)) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {xs, ys} -> mult(xs, ys) end)
  end

  def mult(list1, list2) when is_list(list1) and is_list(list2) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {x, y} -> x * y end)
  end

  def mult(list, y) when is_list(hd(list)) do
    list
    |> Enum.map(&mult(&1, y))
  end

  def mult(list, y) when is_list(list) do
    list
    |> Enum.map(&(&1 * y))
  end

  def mult(x, y) do
    x * y
  end

  @doc """
  Division x divided by y.

  ## Examples
      iex> Broca.NN.division([1, 2, 3], 2)
      [0.5, 1.0, 1.5]

      iex> Broca.NN.division([[1, 2, 3], [1, 2, 3]], 2)
      [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]
  """
  def division(list, y) when is_list(list) do
    list
    |> Enum.map(&division(&1, y))
  end

  def division(x, y) do
    x / y
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
  def transpose(list) when is_list(hd(list)) do
    arr = List.duplicate([], length(hd(list)))

    list
    |> Enum.reverse()
    |> Enum.reduce(arr, fn sub_list, arr ->
      Enum.zip(sub_list, arr)
      |> Enum.map(&([elem(&1, 0)] ++ elem(&1, 1)))
    end)
  end

  def transpose(list) do
    list |> Enum.map(&[&1])
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
  def dot(as, bs) when is_list(hd(as)) and is_list(hd(bs)) do
    # as = if not is_list(hd as), do: [as], else: as
    # bs = if not is_list(hd bs), do: [bs], else: bs
    if length(hd(as)) != length(bs),
      do:
        raise(
          "list should be dot([a x m], [m x b]) but dot([#{length(as)} x #{length(hd(as))}], [#{
            length(bs)
          } x #{length(hd(bs))}]"
        )

    bt = transpose(bs)
    as
    |> Enum.map(fn a -> Enum.map(bt, fn b -> dot(a, b) end) end)
  end

  def dot(as, b) when is_list(hd(as)) do
    as
    |> Enum.map(&dot(&1, b))
  end

  def dot(a, bs) when is_list(hd(bs)) do
    transpose(bs)
    |> Enum.map(&dot(a, &1))
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
    arr = List.duplicate(0, length(hd(list)))

    list
    |> Enum.reduce(arr, fn sub_list, arr ->
      Enum.zip(sub_list, arr)
      |> Enum.map(fn {x, acc} -> x + acc end)
    end)
  end

  def sum(list, _) when is_list(hd(list)) do
    list |> Enum.map(&Enum.sum(&1))
  end

  def sum(list, _) do
    list |> Enum.sum()
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
  def sigmoid(list) when is_list(hd(list)) do
    list |> Enum.map(&sigmoid(&1))
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
  def relu(list) when is_list(hd(list)) do
    list |> Enum.map(&relu(&1))
  end

  def relu(list) do
    list |> Enum.map(&max(0.0, &1))
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
  def softmax(list) when is_list(hd(list)) do
    list |> Enum.map(&softmax(&1))
  end

  def softmax(list) do
    max = Enum.max(list)

    {sum, res} =
      list
      |> Enum.reverse()
      |> Enum.reduce(
        {0, []},
        fn x, {s, r} ->
          ex = :math.exp(x - max)
          {s + ex, [ex] ++ r}
        end
      )

    res |> Enum.map(&(&1 / sum))
  end

  @doc """
  Convert class list to one hot vector.

  ## Examples
      iex> Broca.NN.one_hot(0, 4)
      [1.0, 0.0, 0.0, 0.0, 0.0]

      iex> Broca.NN.one_hot(3, 4)
      [0.0, 0.0, 0.0, 1.0, 0.0]

      iex> Broca.NN.one_hot([0, 1, 2, 3, 4], 4)
      [[1.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0]]
  """
  def one_hot(list, max) when is_list(list) do
    list |> Enum.map(&one_hot(&1, max))
  end

  def one_hot(class, max) do
    0..max |> Enum.map(&if &1 == class, do: 1.0, else: 0.0)
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
      0.510825457099338
  """
  def cross_entropy_error(ys, ts) when is_list(hd(ys)) do
    Enum.zip(ys, ts)
    |> Enum.reduce(0, fn {y, t}, acc -> acc + cross_entropy_error(y, t) end)
    |> Broca.NN.division(length(ys))
  end

  def cross_entropy_error(ys, ts) do
    delta = 1.0e-7
    # IO.inspect(ys)
    # IO.inspect(ts)

    Enum.zip(ys, ts)
    |> Enum.reduce(0, fn {y, t}, acc -> if t == 0, do: acc, else: acc - :math.log(y + delta) end)

    # |> IO.inspect
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
  def argmax(list) when is_list(hd(list)) do
    list |> Enum.map(&argmax(&1))
  end

  def argmax(list) do
    {res, _, _} =
      list
      |> Enum.reduce({-1, -999_999, 0}, fn x, {idx, max, cur} ->
        if x > max, do: {cur, x, cur + 1}, else: {idx, max, cur + 1}
      end)

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
  def filter_mask(list, filter) when is_list(hd(list)) do
    list
    |> Enum.map(&filter_mask(&1, filter))
  end

  def filter_mask(list, filter) do
    list
    |> Enum.map(&if filter.(&1), do: True, else: False)
  end

  @doc """
  Mask the data. The `value` is replaced by the `replaced_value` if `filter` is `True`

  ## Examples
      iex> Broca.NN.mask(True, 4.0)
      0.0

      iex> Broca.NN.mask(False, 4, 0)
      4

      iex> Broca.NN.mask([True, False, True], [1, 2, 4], -1.0)
      [-1.0, 2, -1.0]
  """
  def mask(filter, values) do
    mask(filter, values, 0.0)
  end

  def mask(filter, values, replace_value) when is_list(filter) do
    Enum.zip(filter, values)
    |> Enum.map(fn {f, v} -> mask(f, v, replace_value) end)
  end

  def mask(filter, _, replace_value) when filter == True do
    replace_value
  end

  def mask(_, value, _) do
    value
  end

  @doc """
  Generate zeros with following the structure of `list` given.

  ## Examples
      iex> Broca.NN.zeros_like([])
      []

      iex> Broca.NN.zeros_like([[1], []])
      [[0.0], []]

      iex> Broca.NN.zeros_like([1, 2, 3])
      [0.0, 0.0, 0.0]

      iex> Broca.NN.zeros_like([[1, 2], [3, 4, 5]])
      [[0.0, 0.0], [0.0, 0.0, 0.0]]
  """
  def zeros_like(list) when is_list(hd(list)) do
    list |> Enum.map(&zeros_like(&1))
  end

  def zeros_like(list) do
    List.duplicate(0.0, length(list))
  end

  @doc """
  Get the list of lengthes

  ## Examples
      iex> Broca.NN.shape([1, 2, 3])
      [3]

      iex> Broca.NN.shape([[1, 2], [3, 4], [5, 6]])
      [3, 2]
  """
  def shape(list) do
    shape(list, [])
  end

  def shape(list, res) when is_list(list) do
    shape(hd(list), [length(list)] ++ res)
  end

  def shape(_, res) do
    Enum.reverse(res)
  end
end
