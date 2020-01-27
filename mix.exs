defmodule Broca.MixProject do
  use Mix.Project

  @description """
    Elixir Deep Learning Framework
  """
  def project do
    [
      app: :broca,
      version: "0.1.0",
      elixir: "~> 1.9",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: @description,
      package: package(),
      compilers: [:elixir_make] ++ Mix.compilers()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:earmark, "~> 1.1", only: :dev},
      {:ex_doc, "~> 0.19", only: :dev},
      {:benchfella, "~> 0.3.0", only: :dev},
      {:flow, "~> 0.14.3"},
      {:pelemay, git: "https://github.com/zeam-vm/pelemay.git"},
      {:elixir_make, "~> 0.4", runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["masatam81"],
      licenses: ["MIT"],
      links: %{"Github" => "https://github.com/muramasa8191/broca"}
    ]
  end
end
