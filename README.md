This repository contains code for the simulations reported in the article

Remembering the "When": Hebbian Memory Models for the Time of Past Events, Johanni Brea, Alireza Modirshanechi, Georgios Iatropoulos, Wulfram Gerstner.

To run the simulations, clone the repository, navigate to the `scripts` folder and run the following code in a Julia REPL (tested with Julia version 1.9).

```julia
using Pkg
Pkg.activate()
Pkg.instantiate()

mkdir(joinpath(@__DIR__, "..", "doc"))
mkdir(joinpath(@__DIR__, "..", "data"))

include("sims.jl")
```
