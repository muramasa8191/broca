defmodule Broca.Layers.Convolution do
  @moduledoc """
  Convolution Layer
  """

  defstruct filter_height: 1, filter_width: 1, stride: 1, padding: 0, params: [], grads: []
  
  @doc """
  Helper function to create Convolution
  ## Examples
    iex> input = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    iex> Broca.Layers.Convolution.new(3, 3, [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], [0.0, 0.0])
    %Broca.Layers.Convolution{filter_height: 3, filter_width: 3, params: [weight: [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], bias: [0.0, 0.0]]}

  """
  def new(height, width, weight, bias) do
    %Broca.Layers.Convolution{filter_height: height, filter_width: width, params: [weight: weight, bias: bias]}
  end

  defimpl Layer, for: Broca.Layers.Convolution do
    
    @doc """
    Forward for Convolution

    ## Examples
      iex> input = [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]], \
      [[[26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40], [41, 42, 43, 44, 45], [46, 47, 48, 49, 50]]]]
      iex> conv = Broca.Layers.Convolution.new(3, 3, [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]], [0, 0])
      iex> Layer.forward(conv, input)
      [[[[ 63,  72,  81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]],
       [[[ 63,  72,  81], [108, 117, 126], [153, 162, 171]], [[126, 144, 162], [216, 234, 252], [306, 324, 342]]]]
    """
    def forward(layer, input) do
      out = Broca.NN.matrix_filtering(input, layer.filter_height,
       layer.filter_width, layer.stride, layer.padding, 
       fn list -> Broca.NN.mult(list, layer.params[:weight]) |> Broca.NN.sum(:row) |> Broca.NN.add(layer.params[:bias]) end)
      
      out
      |> Enum.reduce([], fn channel, res -> 
        new_channel = Enum.reduce((hd (hd channel)), [], fn data, acc -> [[data]]++acc end)
        new_channel++res 
        end)
      |> Enum.reverse
    end
    def backward(_layer, dout) do
      dout
    end

    def update(_layer, _optimize_func) do
      :ok
    end
    def batch_update(_layer1, _layer2) do
      :ok
    end
    def get_grads(_layer) do
      :ok
    end
    def gradient_forward(_layer, _x, _name, _idx1, _idx2, _idx3, _diff) do
      :ok
    end
  end
end