defmodule Bench.NN do
  use Benchfella

  @list1 Broca.Random.randn(4320, 100)
  @list2 Broca.Random.randn(100, 4320)
  @data Broca.Random.randn(100, 64, 24, 24)
  @x1 Broca.Random.randn(25, 4320)
  @x2 Broca.Random.randn(4320, 25)
  @small_data Broca.Random.randn(100, 100)
  @add_1d Broca.Random.randn(1, 100)

  bench "Naive transpose" do
    Broca.Naive.NN.transpose(@list1)
  end

  bench "Nif transpose" do
    Broca.Nif.NN.transpose(@list1)
  end

  bench "Naive transpose0231" do
    Broca.Naive.NN.transpose(@data, 0, 2, 3, 1)
  end

  bench "Nif transpose0231" do
    Broca.Nif.NN.transpose(@data, 0, 2, 3, 1)
  end

  bench "Naive transpose0312" do
    Broca.Naive.NN.transpose(@data, 0, 3, 1, 2)
  end

  bench "Nif transpose0312" do
    Broca.Nif.NN.transpose(@data, 0, 3, 1, 2)
  end

  bench "Naive dot" do
    Broca.Naive.NN.dot(@x1, @list1)
  end

  bench "Nif dot" do
    Broca.Nif.NN.dot(@x1, @list1)
  end

  bench "Naive dot_nt" do
    Broca.Naive.NN.dot_nt(@x1, @list2)
  end

  bench "Nif dot_nt" do
    Broca.Nif.NN.dot_nt(@x1, @list2)
  end

  bench "Naive dot_tn" do
    Broca.Naive.NN.dot_tn(@x2, @list1)
  end

  bench "Nif dot_tn" do
    Broca.Nif.NN.dot_tn(@x2, @list1)
  end

  bench "Naive add" do
    Broca.Naive.NN.add(@list1, @list1)
  end

  bench "Nif add" do
    Broca.Nif.NN.add(@list1, @list1)
  end

  bench "Naive add small" do
    Broca.Naive.NN.add(@small_data, @small_data)
  end

  bench "Nif add small" do
    Broca.Nif.NN.add(@small_data, @small_data)
  end

  bench "Naive subtract" do
    Broca.Naive.NN.subtract(@list1, @list1)
  end

  bench "Nif subtract" do
    Broca.Nif.NN.subtract(@list1, @list1)
  end

  bench "Naive subtract small" do
    Broca.Naive.NN.subtract(@small_data, @small_data)
  end

  bench "Nif subtract small" do
    Broca.Nif.NN.subtract(@small_data, @small_data)
  end

end