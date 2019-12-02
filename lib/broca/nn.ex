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

  def mult(list1, list2) when not is_list(hd(list1)) and is_list(hd(list2)) do
    list2
    |> Enum.map(&mult(list1, &1))
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
  def division(list1, list2) when is_list(list1) and is_list(list2) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {sub_list1, sub_list2} -> division(sub_list1, sub_list2) end)
  end

  def division(list, y) when is_list(list) do
    list
    |> Enum.map(&division(&1, y))
  end

  def division(x, y) do
    x / y
  end

  defguard is_2dlist(list) when is_list(hd(list)) and not is_list(hd(hd(list)))
  defguard is_3dlist(list) when is_list(hd(hd(list))) and not is_list(hd(hd(hd(list))))
  defguard is_4dlist(list) when is_list(hd(hd(hd(list))))

  defp _concat(list1, list2, merge \\ True)

  defp _concat(list, nil, _) do
    list
  end

  defp _concat(list1, list2, True) when is_list(hd(list1)) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {sub_list1, sub_list2} -> _concat(sub_list1, sub_list2, True) end)
  end

  defp _concat(list1, list2, False) when is_list(hd(hd(list1))) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {sub_list1, sub_list2} -> _concat(sub_list1, sub_list2, False) end)
  end

  defp _concat(list1, list2, _) do
    Enum.concat(list1, list2)
  end

  @doc """
  Transpose the 4d `list` given

  ## Examples
      iex> Broca.NN.transpose([1, 2, 3])
      [[1], [2], [3]]

      iex> Broca.NN.transpose([[1, 2, 3], [4, 5, 6]])
      [[1, 4], [2, 5], [3, 6]]

      iex> list = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]], \
      [[26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40], [41, 42, 43, 44, 45], [46, 47, 48, 49, 50]]]
      iex> Broca.NN.transpose(list)
      [[[1, 26], [6, 31], [11, 36], [16, 41], [21, 46]],
       [[2, 27], [7, 32], [12, 37], [17, 42], [22, 47]],
       [[3, 28], [8, 33], [13, 38], [18, 43], [23, 48]],
       [[4, 29], [9, 34], [14, 39], [19, 44], [24, 49]],
       [[5, 30], [10, 35], [15, 40], [20, 45], [25, 50]]]

      iex> list = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], \
      [[[25 , 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]], [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47 , 48]]]]
      iex> Broca.NN.transpose(list)
      [[[[1, 25], [13, 37]], [[5, 29], [17, 41]], [[9, 33], [21, 45]]],
      [[[2, 26], [14, 38]], [[6, 30], [18, 42]], [[10, 34], [22, 46]]],
      [[[3, 27], [15, 39]], [[7, 31], [19, 43]], [[11, 35], [23, 47]]],
      [[[4, 28], [16, 40]], [[8, 32], [20, 44]], [[12, 36], [24, 48]]]]

  """
  @spec transpose([number]) :: [number]
  def transpose(list) when is_4dlist(list) do
    arr = List.duplicate([], length(hd(hd(hd(list)))))

    list
    |> Enum.reverse()
    |> Enum.map(fn list2 ->
      Enum.map(Enum.reverse(list2), fn list3 ->
        Enum.reduce(Enum.reverse(list3), arr, fn sub_list, arr ->
          Enum.zip(sub_list, arr)
          |> Enum.map(&([[[elem(&1, 0)]]] ++ elem(&1, 1)))
        end)
      end)
      |> Enum.reduce(
        nil,
        fn channel, acc -> _concat(channel, acc, False) end
      )
    end)
    |> Enum.reduce(
      nil,
      fn channel, acc -> _concat(channel, acc) end
    )
  end

  def transpose(list) when is_3dlist(list) do
    arr = List.duplicate([], length(hd(hd(list))))

    list
    |> Enum.reverse()
    |> Enum.map(fn list2 ->
      Enum.reduce(Enum.reverse(list2), arr, fn sub_list, arr ->
        Enum.zip(sub_list, arr)
        |> Enum.map(&([[elem(&1, 0)]] ++ elem(&1, 1)))
      end)
    end)
    |> Enum.reduce(
      nil,
      fn batch, acc -> _concat(batch, acc) end
    )
  end

  def transpose(list) when is_2dlist(list) do
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
  Transpose axes

  ## Examples
      iex> list = [[[[63, 126], [72, 144], [81, 162]], [[108, 216], [117, 234], [126, 252]], [[153, 306], [162, 324], [171, 342]]], \
      [[[288, 576], [297, 594], [306, 612]], [[333, 666], [342, 684], [351, 702]], [[378, 756], [387, 774], [396, 792]]]]
      iex> Broca.NN.transpose(list, 0, 3, 1, 2)
      [[[[63, 72, 81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]],\
       [[[288, 297, 306], [333, 342, 351], [378, 387, 396]], [[576, 594, 612], [666, 684, 702], [756, 774, 792]]]]

  """
  def transpose(list, 0, 3, 1, 2) do
    list
    |> Enum.map(&transpose(&1, 2, 0, 1))
  end

  def transpose(batch, 2, 0, 1) do
    batch
    |> Enum.reduce(
      List.duplicate([], length(hd(hd(batch)))),
      fn list, acc ->
        data = transpose(list)
        Enum.zip(data, acc) |> Enum.map(fn {val, ac} -> [val] ++ ac end)
      end
    )
    |> Enum.map(&Enum.reverse(&1))
  end

  @doc """
  Reshape the 2D list

  ## Examples
      iex> list = [1, 2, 3, 4]
      iex> Broca.NN.reshape(list, [2, 2])
      [[1, 2], [3, 4]]

      iex> list = [1, 2, 3, 4, 5, 6]
      iex> Broca.NN.reshape(list, [3, 2])
      [[1, 2], [3, 4], [5, 6]]

      iex> list = [1, 2, 3, 4, 5, 6, 7, 8]
      iex> Broca.NN.reshape(list, [2, 2, 2])
      [[[1, 2], [3, 4]],[[5, 6], [7, 8]]]

      iex> list = [[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], [[[[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]]]
      iex> Broca.NN.reshape(list, [2, 2, 2, 2])
      [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
  """
  def reshape(list, dims) do
    list = if is_list(hd(list)), do: List.flatten(list), else: list

    dims
    |> Enum.reverse()
    |> Enum.reduce(list, fn dim, data -> Enum.chunk_every(data, dim) end)
    |> hd
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
  def zeros_like([]) do
    []
  end

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

  @doc """
  Create Shape string

  ## Examples
      iex> Broca.NN.shape_string([[1, 2], [2, 2]])
      "[2, 2]"
  """
  def shape_string([]) do
    "[]"
  end

  def shape_string(list) do
    list_string(shape(list))
  end

  def list_string(list) do
    str =
      list
      |> Enum.reduce(
        "",
        fn dim, str ->
          (str <> Integer.to_string(dim)) <> ", "
        end
      )
      |> String.trim_trailing(", ")

    ("[" <> str) <> "]"
  end

  @doc """
  Add 0.0 `pad_count` times around the `list` given

  ## Examples
      iex> a = [1..10 |> Enum.to_list |> Enum.map(&(&1 / 1.0))]
      iex> Broca.NN.pad(a, 1)
      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

      iex> a = [1..10 |> Enum.to_list |> Enum.map(&(&1 / 1.0))]
      iex> Broca.NN.pad(a, 2)
      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

      iex> Broca.NN.pad([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1)
      [[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1, 2, 3, 0.0],[0.0, 4, 5, 6, 0.0],[0.0, 7, 8, 9, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]]
  """
  def pad(list, pad_count) when is_list(hd(list)) do
    list =
      list
      |> Enum.map(
        &(List.duplicate(0.0, pad_count) ++
            Enum.reverse(List.duplicate(0.0, pad_count) ++ Enum.reverse(&1)))
      )

    List.duplicate(List.duplicate(0.0, length(hd(list))), pad_count) ++
      Enum.reverse(
        List.duplicate(List.duplicate(0.0, length(hd(list))), pad_count) ++ Enum.reverse(list)
      )
  end

  @doc """
  Create filtered list

  ## Examples
      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex>  Broca.NN.matrix_filtering(list, 3, 3)
      [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [2, 3, 4, 7, 8, 9, 12, 13, 14], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[6, 7, 8, 11, 12, 13, 16, 17, 18], [7, 8, 9, 12, 13, 14, 17, 18, 19], [8, 9, 10, 13, 14, 15, 18, 19, 20]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [12, 13, 14, 17, 18, 19, 22, 23, 24], [13, 14, 15, 18, 19, 20, 23, 24, 25]]]]

      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex>  Broca.NN.matrix_filtering(list, 3, 3, 2)
      [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [13, 14, 15, 18, 19, 20, 23, 24, 25]]]]

      iex> list = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
      iex>  Broca.NN.matrix_filtering(list, 3, 3, 1, 1)
      [[[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]]]

      iex> list = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
      iex>  Broca.NN.matrix_filtering(list, 3, 3, 1, 1)
      [[[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]],
       [[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]]]

      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex> Broca.NN.matrix_filtering(list, 3, 3, 1, 0, fn l -> Enum.max(l) end)
      [[[13, 14, 15],
       [18, 19, 20],
       [23, 24, 25]]]

      iex> tensor = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]], \
      [[[28, 29, 30], [31, 32, 33], [34, 35, 36]], [[37, 38, 39], [40, 41, 42], [43, 44, 45]], [[46, 47, 48], [49, 50, 51], [52, 53, 54]]]]
      iex> Broca.NN.matrix_filtering(tensor, 2, 2, 1, 0, fn x -> x end)
      [[[[1, 2, 4, 5, 10, 11, 13, 14, 19, 20, 22, 23], [2, 3, 5, 6, 11, 12, 14, 15, 20, 21, 23, 24]],
        [[4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26], [5, 6, 8, 9, 14, 15, 17, 18, 23, 24, 26, 27]]],
       [[[28, 29, 31, 32, 37, 38, 40, 41, 46, 47, 49, 50], [29, 30, 32, 33, 38, 39, 41, 42, 47, 48, 50, 51]],
        [[31, 32, 34, 35, 40, 41, 43, 44, 49, 50, 52, 53], [32, 33, 35, 36, 41, 42, 44, 45, 50, 51, 53, 54]]]]

  """
  def matrix_filtering(
        list,
        filter_height,
        filter_width,
        stride \\ 1,
        padding \\ 0,
        map_func \\ fn list -> list end,
        type \\ :merge
      )

  def matrix_filtering(list, filter_height, filter_width, stride, padding, map_func, type)
      when is_4dlist(list) do
    Enum.map(
      list,
      &matrix_filtering(&1, filter_height, filter_width, stride, padding, map_func, type)
    )
  end

  def matrix_filtering(list, filter_height, filter_width, stride, padding, map_func, :merge)
      when is_3dlist(list) do
    Enum.reduce(
      list,
      [],
      &_concat(&2, _matrix_filtering(&1, filter_height, filter_width, stride, padding, map_func))
    )
  end

  def matrix_filtering(list, filter_height, filter_width, stride, padding, map_func, _)
      when is_3dlist(list) do
    Enum.map(
      list,
      &_matrix_filtering(&1, filter_height, filter_width, stride, padding, map_func)
    )
  end

  defp _matrix_filtering(list, filter_height, filter_width, stride, padding, map_func) do
    list = if padding == 0, do: list, else: pad(list, padding)
    org_h = length(list)
    org_w = length(hd(list))
    out_h = div(org_h - filter_height, stride) + 1
    out_w = div(org_w - filter_width, stride) + 1

    for y <- for(i <- 0..(out_h - 1), do: i * stride) |> Enum.filter(&(&1 < org_h)) do
      for x <- for(i <- 0..(out_w - 1), do: i * stride) |> Enum.filter(&(&1 < org_w)) do
        list
        |> Enum.drop(y)
        |> Enum.take(filter_height)
        |> Enum.map(&(Enum.drop(&1, x) |> Enum.take(filter_width)))
        |> List.flatten()
        |> map_func.()
      end
    end
  end

  def for_each(list, func) when is_list(list) do
    Enum.map(list, &for_each(&1, func))
  end

  def for_each(val, func) do
    func.(val)
  end
end
