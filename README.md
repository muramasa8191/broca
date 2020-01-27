[![CircleCI](https://circleci.com/gh/muramasa8191/broca.svg?style=svg&circle-token=8d5930c563189e3b9a997369c705f15841d948e6)](https://circleci.com/gh/muramasa8191/broca)
# Broca

Deep Learning Framework written in Elixir.

# Installation

Broca is not available in Hex, the package can be installed
by adding `broca` with git to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:broca, git: "https://github.com/muramasa8191/broca.git"}
  ]
end
```

## Dependency

There are some dependencies to run Broca.

### OpenBLAS
For the first version, you need to install OpenBLAS.

#### MacOSX

Using brew, you can install like below.
```console
$ brew install openblas
```

#### CentOS

Please install via yum with EPEL.

```console
$ sudo yum install -y epel-release
$ sudo yum install -y openblas-devel
```

#### Debian

Please install via apt-get

```console
$ sudo apt-get install libopenblas-dev
```

#### Windows

TBA

### Elixir libraries

Just kick the commands below.

```console
$ mix deps.get
$ mix deps.compile
```

# Usage

- Training

1. Create the layer list by using factory functions in `Broca.Layers` module

```elixir
iex(1)> model = [
...(1)>   Layers.convolution2d(5, 5, 1, 30, 1, 0),
...(1)>   Layers.relu(),
...(1)>   Layers.max_pooling2d(2, 2, 2, 0),
...(1)>   Layers.affine(4320, 100),
...(1)>   Layers.relu(),
...(1)>   Layers.affine(100, 10),
...(1)>   Layers.softmax()
...(1)> ]
]
```

2. Create training setting

```elixir
iex(2)> setting = Broca.Trainer.setting(1, 400, :adam, 0.001, 4)
[
  epochs: 1,
  batch_size: 400,
  optimizer_type: :adam,
  learning_rate: 0.001,
  pallarel_size: 4
]
```

3. Create DataLoader. Factory function for MNIST and CIFAR10 is available.

```elixir
iex(3)> train_dataloader = Broca.DataLoaders.create(:mnist_train)
iex(4)> test_dataloader = Broca.DataLoaders.create(:mnist_validation)
```

4. Feed them to the Trainer

```elixir
iex(5)> Broca.Trainer.train(model, setting, train_dataloader, test_dataloader)
```

## Layers

Folloing layers are already implemented Althought the backward of Convolution2d is too slow.

- Affine
- Convolution2D
- MaxPooling2D
- Dropout

## Activations

Following activation layers are ready.
- ReLU
- Softmax

## Optimizers

Following optimizers are available.

- Adam(:adam)
- SGD(:sgd)

Set the Atom on the setting for Trainer


Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/broca](https://hexdocs.pm/broca).

