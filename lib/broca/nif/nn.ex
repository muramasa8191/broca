defmodule Broca.Nif.NN do
  @moduledoc """
  Module to handle `list` and `list of lists` for Neural Network.
  """

  @on_load :load_nifs

  def load_nifs do
    priv_dir =
      case :code.priv_dir(__MODULE__) do
        {:error, _} ->
          ebin_dir = :code.which(__MODULE__) |> :filename.dirname()
          app_path = :filename.dirname(ebin_dir)
          :filename.join(app_path, "priv")

        path ->
          path
      end

    case :erlang.load_nif(:filename.join(priv_dir, "nn"), 0) do
      :ok ->
        :ok

      {:error, {:load_failed, reason}} ->
        IO.warn("Error loading NIF #{reason}")
        :ok
    end
  end

  def nif_transpose2d(_matrix) do
    raise "NIF nif_transpose2d/1 not implemented"
  end

  def nif_transpose4d(_tensor, _axis0, _axis1, _axis2, _axis3) do
    raise "NIF nif_transpose4d/5 not implemented"
  end

  def nif_dot(_list1, _list2) do
    raise "NIF nif_dot/2 not implemented"
  end

  def nif_dot_nt(_list1, _list2) do
    raise "NIF nif_dot_nt/2 not implemented"
  end

  def nif_dot_tn(_list1, _list2) do
    raise "NIF nif_dot_tn/2 not implemented"
  end

  def nif_add(_list1, _list2) do
    raise "NIF nif_add/2 not implemented"
  end

  def nif_subtract(_list1, _list2) do
    raise "NIF nif_subtract/2 not implemented"
  end

  defguard is_2dlist(list) when is_list(hd(list)) and not is_list(hd(hd(list)))
  defguard is_3dlist(list) when is_list(hd(hd(list))) and not is_list(hd(hd(hd(list))))
  defguard is_4dlist(list) when is_list(hd(hd(hd(list))))

  @doc """
  Add `list2` to `list1` 

  ## Examples
      iex> Broca.Nif.NN.add([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
      [[6.0, 8.0], [8.0, 10.0]]

      iex> Broca.Nif.NN.add([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
      [[6.0, 8.0], [10.0, 12.0]]

      iex> Broca.Nif.NN.add([[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]])
      [[[6.0, 8.0], [10.0, 12.0]]]
  """
  @spec add([[number]], [[number]]) :: [[number]]
  def add(list1, list2) do
    size1 = data_size(list1)
    size2 = data_size(list2)

    if size1 == size2 do
      if size1 >= 10000 do
        nif_add(list2, list1)
      else
        Broca.Naive.NN.add(list1, list2)
      end
    else
      Broca.Naive.NN.add(list1, list2)
    end
  end

  @doc """
  Subtract `arg2` from `arg1`

  ## Examples
      iex> Broca.Nif.NN.subtract(10.0, 7.0)
      3.0

      iex> Broca.Nif.NN.subtract([1.0, 2.0, 3.0], 1.0)
      [0.0, 1.0, 2.0]

      iex> Broca.Nif.NN.subtract(1.0, [1.0, 2.0, 3.0])
      [0.0, -1.0, -2.0]

      iex> Broca.Nif.NN.subtract([3.0, 4.0, 5.0], [1.0, 2.0, 3.0])
      [2.0, 2.0, 2.0]

      iex> Broca.Nif.NN.subtract([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]], [1.0, 2.0, 1.0])
      [[0.0, 0.0, 2.0], [10.0, 10.0, 12.0]]

      iex> Broca.Nif.NN.subtract([[99.0, 98.0], [97.0, 96.0]], [[1.0, 2.0], [3.0, 4.0]])
      [[98.0, 96.0], [94.0, 92.0]]
  """
  def subtract(list1, list2) when is_list(list1) and is_list(list2) do
    size1 = data_size(list1)
    size2 = data_size(list2)

    if size1 == size2 do
      if size1 >= 10000 do
        nif_subtract(list1, list2)
      else
        Broca.Naive.NN.subtract(list1, list2)
      end
    else
      Broca.Naive.NN.subtract(list1, list2)
    end
  end

  def subtract(list1, list2) do
    Broca.Naive.NN.subtract(list1, list2)
  end

  @doc """
  Vector multiplication

  """
  def mult(list1, list2) do
    Broca.Naive.NN.mult(list1, list2)
  end

  @doc """
  Division x divided by y.

  ## Examples
      iex> Broca.Nif.NN.division([1, 2, 3], 2)
      [0.5, 1.0, 1.5]

      iex> Broca.Nif.NN.division([[1, 2, 3], [1, 2, 3]], 2)
      [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]
  """
  def division(list1, list2) do
    Broca.Naive.NN.division(list1, list2)
  end

  defp concat(list1, list2, merge \\ true)

  defp concat(nil, list, _) do
    list
  end

  defp concat(list, nil, _) do
    list
  end

  defp concat(list1, list2, true) when is_list(hd(list2)) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {sub_list1, sub_list2} -> concat(sub_list1, sub_list2, true) end)
  end

  defp concat(list1, list2, false) when is_list(hd(hd(list2))) do
    Enum.zip(list1, list2)
    |> Enum.map(fn {sub_list1, sub_list2} -> concat(sub_list1, sub_list2, false) end)
  end

  defp concat(list1, list2, _) do
    Enum.concat(list1, list2)
  end

  @doc """
  Transpose the 4d `list` given

  ## Examples
      iex> Broca.Nif.NN.transpose([1.0, 2.0, 3.0])
      [[1.0], [2.0], [3.0]]

      iex> Broca.Nif.NN.transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

      iex> list = [[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0],\
      [11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0],\
      [21.0, 22.0, 23.0, 24.0, 25.0]], [[26.0, 27.0, 28.0, 29.0, 30.0], \
      [31.0, 32.0, 33.0, 34.0, 35.0], [36.0, 37.0, 38.0, 39.0, 40.0], \
      [41.0, 42.0, 43.0, 44.0, 45.0], [46.0, 47.0, 48.0, 49.0, 50.0]]]
      iex> Broca.Nif.NN.transpose(list)
      [[[1.0, 26.0], [6.0, 31.0], [11.0, 36.0], [16.0, 41.0], [21.0, 46.0]],
       [[2.0, 27.0], [7.0, 32.0], [12.0, 37.0], [17.0, 42.0], [22.0, 47.0]],
       [[3.0, 28.0], [8.0, 33.0], [13.0, 38.0], [18.0, 43.0], [23.0, 48.0]],
       [[4.0, 29.0], [9.0, 34.0], [14.0, 39.0], [19.0, 44.0], [24.0, 49.0]],
       [[5.0, 30.0], [10.0, 35.0], [15.0, 40.0], [20.0, 45.0], [25.0, 50.0]]]

      iex> list = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], \
      [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]], \
      [[[25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0], [33.0, 34.0, 35.0, 36.0]], \
      [[37.0, 38.0, 39.0, 40.0], [41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0]]]]
      iex> Broca.Nif.NN.transpose(list)
      [[[[1.0, 25.0], [13.0, 37.0]], [[5.0, 29.0], [17.0, 41.0]], [[9.0, 33.0], [21.0, 45.0]]],
      [[[2.0, 26.0], [14.0, 38.0]], [[6.0, 30.0], [18.0, 42.0]], [[10.0, 34.0], [22.0, 46.0]]],
      [[[3.0, 27.0], [15.0, 39.0]], [[7.0, 31.0], [19.0, 43.0]], [[11.0, 35.0], [23.0, 47.0]]],
      [[[4.0, 28.0], [16.0, 40.0]], [[8.0, 32.0], [20.0, 44.0]], [[12.0, 36.0], [24.0, 48.0]]]]

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
        fn channel, acc -> concat(channel, acc, false) end
      )
    end)
    |> Enum.reduce(
      nil,
      fn channel, acc -> concat(channel, acc) end
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
      fn batch, acc -> concat(batch, acc) end
    )
  end

  def transpose(list) when is_2dlist(list) do
    nif_transpose2d(list)
  end

  def transpose(list) do
    list |> Enum.map(&[&1])
  end

  @doc """
  Transpose axes

  ## Examples
      iex> list = [[[[63.0, 126.0], [72.0, 144.0], [81.0, 162.0]], [[108.0, 216.0], \
      [117.0, 234.0], [126.0, 252.0]], [[153.0, 306.0], [162.0, 324.0], [171.0, 342.0]]], \
      [[[288.0, 576.0], [297.0, 594.0], [306.0, 612.0]], [[333.0, 666.0], [342.0, 684.0], \
      [351.0, 702.0]], [[378.0, 756.0], [387.0, 774.0], [396.0, 792.0]]]]
      iex> Broca.Nif.NN.transpose(list, 0, 3, 1, 2)
      [[[[63.0, 72.0, 81.0], [108.0, 117.0, 126.0], [153.0, 162.0, 171.0]], [[126.0, 144.0, 162.0], [216.0, 234.0, 252.0], [306.0, 324.0, 342.0]]],\
       [[[288.0, 297.0, 306.0], [333.0, 342.0, 351.0], [378.0, 387.0, 396.0]], [[576.0, 594.0, 612.0], [666.0, 684.0, 702.0], [756.0, 774.0, 792.0]]]]

      iex> list = [[[[0.0,  1.0], [2.0,  3.0], [4.0,  5.0]], [[6.0,  7.0], [8.0,  9.0], [10.0, 11.0]], [[12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]], \
      [[[18.0, 19.0], [20.0, 21.0], [22.0, 23.0]], [[24.0, 25.0], [26.0, 27.0], [28.0, 29.0]], [[30.0, 31.0], [32.0, 33.0], [34.0, 35.0]]], \
      [[[36.0, 37.0], [38.0, 39.0], [40.0, 41.0]], [[42.0, 43.0], [44.0, 45.0], [46.0, 47.0]], [[48.0, 49.0], [50.0, 51.0], [52.0, 53.0]]]]
      iex> Broca.Naive.NN.transpose(list, 0, 2, 3, 1)
      [[[[0.0,  6.0, 12.0], [1.0,  7.0, 13.0]], [[2.0,  8.0, 14.0], [3.0,  9.0, 15.0]], [[4.0, 10.0, 16.0], [5.0, 11.0, 17.0]]], 
      [[[18.0, 24.0, 30.0], [19.0, 25.0, 31.0]], [[20.0, 26.0, 32.0], [21.0, 27.0, 33.0]], [[22.0, 28.0, 34.0], [23.0, 29.0, 35.0]]], 
      [[[36.0, 42.0, 48.0], [37.0, 43.0, 49.0]], [[38.0, 44.0, 50.0], [39.0, 45.0, 51.0]], [[40.0, 46.0, 52.0], [41.0, 47.0, 53.0]]]]
  """
  def transpose(list, axis0, axis1, axis2, axis3) do
    nif_transpose4d(list, axis0, axis1, axis2, axis3)
  end

  @doc """
  Reshape the 2D list

  ## Examples
      iex> list = [1, 2, 3, 4]
      iex> Broca.Nif.NN.reshape(list, [2, 2])
      [[1, 2], [3, 4]]

      iex> list = [1, 2, 3, 4, 5, 6]
      iex> Broca.Nif.NN.reshape(list, [3, 2])
      [[1, 2], [3, 4], [5, 6]]

      iex> list = [1, 2, 3, 4, 5, 6, 7, 8]
      iex> Broca.Nif.NN.reshape(list, [2, 2, 2])
      [[[1, 2], [3, 4]],[[5, 6], [7, 8]]]

      iex> list = [[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], [[[[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]]]
      iex> Broca.Nif.NN.reshape(list, [2, 2, 2, 2])
      [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
  """
  def reshape(list, dims) do
    Broca.Naive.NN.reshape(list, dims)
  end

  def dot_and_add(list, list_dot, list_add) do
    Broca.Naive.NN.dot_and_add(list, list_dot, list_add)
  end

  def dot_nt(list1, list2) do
    nif_dot_nt(list1, list2)
  end

  def dot_tn(list1, list2) do
    nif_dot_tn(list1, list2)
  end

  @doc """
  Dot product

  ## Examples
      iex> a = [[1.0, 2.0, 3.0]]
      iex> b = [[4.0], [5.0], [6.0]]
      iex> Broca.Nif.NN.dot(a, b)
      [[32.0]]

      iex> a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
      iex> b = [[7.0], [8.0]]
      iex> Broca.Nif.NN.dot(a, b)
      [[23.0], [53.0], [83.0]]

      iex> a = [[1.0, 2.0], [3.0, 4.0]]
      iex> b = [[5.0, 6.0], [7.0, 8.0]]
      iex> Broca.Nif.NN.dot(a, b)
      [[19.0, 22.0], [43.0, 50.0]]

      iex> a = [[1.0, 2.0]]
      iex> b = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
      iex> Broca.Nif.NN.dot(a, b)
      [[5.0, 11.0, 17.0]]
  """
  @spec dot([[number]], [[number]]) :: [[number]]
  @spec dot([[number]], [number]) :: [number]
  @spec dot([number], [[number]]) :: [number]
  @spec dot([number], [number]) :: number
  def dot(list1, list2) do
    nif_dot(list1, list2)
  end

  @doc """
  Sum the `list`

  ## Examples
      iex> Broca.Nif.NN.sum([1.0, 2.0, 3.0])
      6.0

      iex> Broca.Nif.NN.sum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], :row)
      [6.0, 15.0]

      iex> Broca.Nif.NN.sum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], :col)
      [12.0, 15.0, 18.0]
  """
  def sum(list) do
    Broca.Naive.NN.sum(list, :row)
  end

  def sum(list, axis) do
    Broca.Naive.NN.sum(list, axis)
  end

  @doc """
  Sigmoid function

  ## Examples
      iex> Broca.Nif.NN.sigmoid([-1.0, 1.0, 2.0])
      [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]

      iex> Broca.Nif.NN.sigmoid([[-1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]])
      [[0.2689414213699951, 0.7310585786300049, 0.8807970779778823],
       [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]]
  """
  @spec sigmoid([[number]]) :: [[number]]
  @spec sigmoid([number]) :: [number]
  def sigmoid(list) do
    Broca.Naive.NN.sigmoid(list)
  end

  @doc """
  ReLU function

  ## Examples
      iex> Broca.Nif.NN.relu([-1.0, 0.0, 1.0, 2.0])
      [0.0, 0.0, 1.0, 2.0]

      iex> Broca.Nif.NN.relu([[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0]])
      [[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 1.0, 2.0]]
  """
  @spec relu([[number]]) :: [[number]]
  @spec relu([number]) :: [number]
  def relu(list) do
    Broca.Naive.NN.relu(list)
  end

  @doc """
  Softmax function

  ## Examples
      iex> Broca.Nif.NN.softmax([0.3, 2.9, 4.0])
      [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]

      iex> Broca.Nif.NN.softmax([[0.3, 2.9, 4.0], [0.3, 2.9, 4.0], [0.3, 2.9, 4.0]])
      [[0.01821127329554753, 0.24519181293507392, 0.7365969137693786],
       [0.01821127329554753, 0.24519181293507392, 0.7365969137693786],
       [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]]
  """
  @spec softmax([[number]]) :: [[number]]
  @spec softmax([number]) :: [number]
  def softmax(list) do
    Broca.Naive.NN.softmax(list)
  end

  @doc """
  Convert class list to one hot vector.

  ## Examples
      iex> Broca.Nif.NN.one_hot(0, 4)
      [1.0, 0.0, 0.0, 0.0, 0.0]

      iex> Broca.Nif.NN.one_hot(3, 4)
      [0.0, 0.0, 0.0, 1.0, 0.0]

      iex> Broca.Nif.NN.one_hot([0, 1, 2, 3, 4], 4)
      [[1.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 1.0]]
  """
  def one_hot(list, max) do
    Broca.Naive.NN.one_hot(list, max)
  end

  @doc """
  Calculate cross entropy error.

  ## Examples
      iex> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
      iex> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
      iex> Broca.Nif.NN.cross_entropy_error(y, t)
      0.51082545709933802
    
      iex> t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
      iex> y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]
      iex> Broca.Nif.NN.cross_entropy_error(y, t)
      0.510825457099338
  """
  def cross_entropy_error(ys, ts) do
    Broca.Naive.NN.cross_entropy_error(ys, ts)
  end

  @doc """
  Get the index of maximum value in the list

  ## Examples
      iex> Broca.Nif.NN.argmax([5.0, 1.0, 2.0, 3.0])
      0

      iex> Broca.Nif.NN.argmax([1.0, 4.0, 2.0, 3.0])
      1

      iex> Broca.Nif.NN.argmax([[5.0, 1.0, 2.0, 3.0], [1.0, 4.0, 2.0, 3.0]])
      [0, 1]
  """
  def argmax(list) do
    Broca.Naive.NN.argmax(list)
  end

  @doc """
  Mask the list with the filter given

  ## Examples
      iex> Broca.Nif.NN.filter_mask([-1.0, 0.2, 1.0, -0.3], fn x -> x <= 0 end)
      [true, false, false, true]

      iex> Broca.Nif.NN.filter_mask([[-1.0, 0.2, 1.0, -0.3], [-1.0, 0.2, 1.0, -0.3]], fn x -> x <= 0 end)
      [[true, false, false, true], [true, false, false, true]]
  """
  def filter_mask(list, filter) do
    Broca.Naive.NN.filter_mask(list, filter)
  end

  @doc """
  Mask the data. The `value` is replaced by the `replaced_value` if `filter` is `true`

  ## Examples
      iex> Broca.Nif.NN.mask(true, 4.0)
      0.0

      iex> Broca.Nif.NN.mask(false, 4, 0)
      4

      iex> Broca.Nif.NN.mask([true, false, true], [1, 2, 4], -1.0)
      [-1.0, 2, -1.0]
  """
  def mask(filter, values) do
    Broca.Naive.NN.mask(filter, values)
  end

  def mask(filter, values, replace_value) do
    Broca.Naive.NN.mask(filter, values, replace_value)
  end

  @doc """
  Generate zeros with following the structure of `list` given.

  ## Examples
      iex> Broca.Nif.NN.zeros_like([])
      []

      iex> Broca.Nif.NN.zeros_like([[1], []])
      [[0.0], []]

      iex> Broca.Nif.NN.zeros_like([1, 2, 3])
      [0.0, 0.0, 0.0]

      iex> Broca.Nif.NN.zeros_like([[1, 2], [3, 4, 5]])
      [[0.0, 0.0], [0.0, 0.0, 0.0]]
  """
  def zeros_like(list) do
    Broca.Naive.NN.zeros_like(list)
  end

  @doc """
  Get the list of lengthes

  ## Examples
      iex> Broca.Nif.NN.shape([1, 2, 3])
      [3]

      iex> Broca.Nif.NN.shape([[1, 2], [3, 4], [5, 6]])
      [3, 2]
  """
  def shape(list) do
    Broca.Naive.NN.shape(list)
  end

  @doc """
  The shape of the list
  This expects the list of the lists at same sizes

  ## Examples
      iex> Broca.Nif.NN.data_size([[[1, 1, 1]], [[1, 1, 1]]])
      6
  """
  def data_size(list) do
    Broca.Naive.NN.data_size(list)
  end

  @doc """
  Create Shape string

  ## Examples
      iex> Broca.Nif.NN.shape_string([[1, 2], [2, 2]])
      "[2, 2]"
  """
  def shape_string(list) do
    Broca.Naive.NN.shape_string(list)
  end

  @doc """
  Add 0.0 `pad_count` times around the `list` given

  ## Examples
      iex> a = [1..10 |> Enum.to_list |> Enum.map(&(&1 / 1.0))]
      iex> Broca.Nif.NN.pad(a, 1)
      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

      iex> a = [1..10 |> Enum.to_list |> Enum.map(&(&1 / 1.0))]
      iex> Broca.Nif.NN.pad(a, 2)
      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

      iex> Broca.Nif.NN.pad([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1)
      [[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1, 2, 3, 0.0],[0.0, 4, 5, 6, 0.0],[0.0, 7, 8, 9, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]]
  """
  def pad(list, pad_count) do
    Broca.Naive.NN.pad(list, pad_count)
  end

  @doc """
  Create filtered list

  ## Examples
      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex>  Broca.Nif.NN.matrix_filtering(list, 3, 3)
      [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [2, 3, 4, 7, 8, 9, 12, 13, 14], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[6, 7, 8, 11, 12, 13, 16, 17, 18], [7, 8, 9, 12, 13, 14, 17, 18, 19], [8, 9, 10, 13, 14, 15, 18, 19, 20]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [12, 13, 14, 17, 18, 19, 22, 23, 24], [13, 14, 15, 18, 19, 20, 23, 24, 25]]]]

      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex>  Broca.Nif.NN.matrix_filtering(list, 3, 3, 2)
      [[[[1, 2, 3, 6, 7, 8, 11, 12, 13], [3, 4, 5, 8, 9, 10, 13, 14, 15]],
       [[11, 12, 13, 16, 17, 18, 21, 22, 23], [13, 14, 15, 18, 19, 20, 23, 24, 25]]]]

      iex> list = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
      iex>  Broca.Nif.NN.matrix_filtering(list, 3, 3, 1, 1)
      [[[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]]]

      iex> list = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
      iex>  Broca.Nif.NN.matrix_filtering(list, 3, 3, 1, 1)
      [[[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]],
       [[[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0], [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
       [[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0]],
       [[0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0]]]]

      iex> list = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]
      iex> Broca.Nif.NN.matrix_filtering(list, 3, 3, 1, 0, fn l -> Enum.max(l) end)
      [[[13, 14, 15],
       [18, 19, 20],
       [23, 24, 25]]]

      iex> tensor = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]], \
      [[[28, 29, 30], [31, 32, 33], [34, 35, 36]], [[37, 38, 39], [40, 41, 42], [43, 44, 45]], [[46, 47, 48], [49, 50, 51], [52, 53, 54]]]]
      iex> Broca.Nif.NN.matrix_filtering(tensor, 2, 2, 1, 0, fn x -> x end)
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
      ) do
    Broca.Naive.NN.matrix_filtering(
      list,
      filter_height,
      filter_width,
      stride,
      padding,
      map_func,
      type
    )
  end

  def for_each(list, func) do
    Broca.Naive.NN.for_each(list, func)
  end
end
