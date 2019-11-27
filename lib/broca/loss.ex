defprotocol Loss do
  def loss(layer, y, t)
  def backward(layer, t)
end

defmodule Broca.Losses.CrossEntropyError do
  defstruct val: []

  defimpl Inspect, for: Broca.Losses.CrossEntropyError do
    def inspect(cross_entropy_error, _) do
      "CrossEntropyError: val=#{Broca.NN.shape_string(cross_entropy_error.val)}"
    end
  end

  @moduledoc """
  Cross Entropy Error
  """
  defimpl Loss, for: Broca.Losses.CrossEntropyError do
    @doc """
    Loss

    ## Examples
        iex> Loss.loss(%Broca.Losses.CrossEntropyError{}, [[0.1, 0.4, 0.5]], [[0, 0, 1]])
        0.6931469805599654
    """
    def loss(_, y, t) do
      Broca.NN.cross_entropy_error(y, t)
    end

    @doc """
    Backward

    ## Examples
      iex> Loss.backward(%Broca.Losses.CrossEntropyError{}, [0, 0, 1])
      [0, 0, 1]
    """
    def backward(_, t) do
      t
    end
  end
end

defmodule Broca.Losses.SoftmaxWithLoss do
  defstruct y: nil, t: nil

  defimpl Loss, for: Broca.Losses.SoftmaxWithLoss do
    @doc """
    Calculate loss and update the structure to be returned.

    ## Examples
        iex> t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        iex> x = [[0.5, 0.25, 0.30, 0.0, 0.25, 0.5, 0.0, 0.5, 0.0, 0.0],[0.6, 0.30, 0.36, 0.0, 0.30, 0.6, 0.0, 0.6, 0.0, 0.0]]
        iex> {s, dout} = Loss.loss(%Broca.Losses.SoftmaxWithLoss{}, x, t)
        iex> y2 = s.y |> Enum.map(&Enum.map(&1, fn x -> Float.floor(x, 15) end))
        iex> {%Broca.Losses.SoftmaxWithLoss{s | y: y2}, dout}
        {%Broca.Losses.SoftmaxWithLoss{y: [[0.128164789848038, 0.099814838695834, 0.104932454910366, 0.077735874538462, 0.099814838695834, 0.128164789848038, 0.077735874538462, 0.128164789848038, 0.077735874538462, 0.077735874538462],
         [0.133985200869365, 0.099258678105725, 0.105396491974441, 0.073532637301503, 0.099258678105725, 0.133985200869365, 0.073532637301503, 0.133985200869365, 0.073532637301503, 0.073532637301503]], t: [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]},
         2.2522312235012825}
    """
    def loss(_, x, t) do
      y = Broca.NN.softmax(x)
      loss = Broca.NN.cross_entropy_error(y, t)

      {%Broca.Losses.SoftmaxWithLoss{y: y, t: t}, loss}
    end

    @doc """
    Backward softmax and cross_entropy

    ## Examples
      iex> Loss.backward(%Broca.Losses.SoftmaxWithLoss{y: [[0.12816478984803883, 0.09981483869583468, 0.10493245491036612, 0.07773587453846202, 0.09981483869583468, 0.12816478984803883, 0.07773587453846202, 0.12816478984803883, 0.07773587453846202, 0.07773587453846202], \
         [0.13398520086936513, 0.09925867810572563, 0.1053964919744412, 0.07353263730150306, 0.09925867810572563, 0.13398520086936513, 0.07353263730150306, 0.13398520086936513, 0.07353263730150306, 0.07353263730150306]], t: [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]}, 1)
      [[0.06408239492401942, 0.04990741934791734, -0.44753377254481697, 0.03886793726923101, 0.04990741934791734, 0.06408239492401942, 0.03886793726923101, 0.06408239492401942, 0.03886793726923101, 0.03886793726923101],
       [0.06699260043468257, 0.04962933905286281, -0.4473017540127794, 0.03676631865075153, 0.04962933905286281, 0.06699260043468257, 0.03676631865075153, 0.06699260043468257, 0.03676631865075153, 0.03676631865075153]]
    """
    def backward(layer, _) do
      Broca.NN.subtract(layer.y, layer.t) |> Broca.NN.division(length(layer.y))
    end
  end
end
