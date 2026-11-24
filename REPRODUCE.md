# Soft decision tree induction: node, rate and data parallelism

This file describes how to build and run the three parallel implementations of soft decision tree induction (MPI) used in the experiments of the accompanying paper.

## Layout

The three implementations differ only in `SoftDecisionNode`. Everything else (`SoftDecisionTree`, `SoftDecisionTreeParameter`, the driver `src/main.cpp`) is shared.

| Scheme | Parallelism | Source |
|---|---|---|
| NF | node farm: one manager and p - 1 workers, one tree node per task | `src/Model/DecisionTree/Parallel/NF/` |
| RP | rate parallel: the n = 32 learning rates are divided among the p processes | `src/Model/DecisionTree/Parallel/RP/` |
| DP | data parallel: the training set is divided among the p processes, parameters are averaged after every epoch | `src/Model/DecisionTree/Parallel/DP/` |

With p = 1, RP and DP are the plain sequential algorithm. NF with p = 1 runs the same task encoding in a single process.

## Build

Install the dependencies listed in `conanfile.py` with Conan (dictionary, sampling, data_structure, util, math, openmpi) and pass the generated CMake toolchain file to CMake. The scheme is chosen when the project is configured:

```bash
cmake -S . -B build-nf -DCMAKE_TOOLCHAIN_FILE=<conan_toolchain.cmake> -DSCHEME=NF
cmake --build build-nf --target Main
```

`SCHEME` is `NF` (default), `RP` or `DP`. A separate build directory is needed for each scheme.

## Run

The seven data sets are taken from the UCI Machine Learning Repository and are expected in `./datasets/<name>.data`, with the names `dermatology`, `iris`, `bupa`, `optdigits`, `pendigits`, `wave` and `yeast`.

```bash
mpirun -np 4 ./Main                # all seven data sets
mpirun -np 4 ./Main iris bupa      # only the named data sets
```

The number of MPI processes is p.

| Scheme | Allowed p |
|---|---|
| NF | any p >= 1 (rank 0 is the manager and does no fitting, so p >= 2 gives p - 1 workers) |
| RP | p must divide n = 32 (1, 2, 4, 8, 16, 32) |
| DP | any p >= 1 |

## Settings

| Setting | Value |
|---|---|
| Learning rates | n = 32, eta_i = 10 / 2^i |
| Decay of eta per epoch | 0.99 |
| Momentum | 0 |
| Seed | 1 |
| Validation fraction | 0.2 (80 / 20 split of the training set) |
| Epochs | 100 for dermatology, iris, bupa and yeast; 10 for optdigits, pendigits and wave |

## Output

For every data set the driver prints the error rate and the induction time in seconds. The error rate is measured on the data set the tree was induced from; it is not a held-out estimate. With NF only rank 0 holds the induced tree.

## Reproducibility

The tree induced by NF does not depend on p. RP and DP do depend on p: RP changes the order in which the training instances are visited, DP averages the parameters of p partial models after every epoch. The tree is therefore not identical across p for these two schemes.

## Results in the paper

The results in the paper were produced on the Özyeğin University cluster (Linux), on one exclusive node, with `mpirun -np p ./Main`. The sources that were run there are the ones in this repository:

| Scheme | Source | Driver |
|---|---|---|
| NF | `src/Model/DecisionTree/Parallel/NF/` | `experiments/main_NF_RP.cpp` |
| RP | `src/Model/DecisionTree/Parallel/RP/` | `experiments/main_NF_RP.cpp` |
| DP | `src/Model/DecisionTree/Parallel/DP/` | `experiments/main_DP.cpp` |

In `RP`, the processes other than rank 0 send the parameters of their best rate (`bestNode`) to rank 0. The first version of this file (commit b860de4, taken unchanged from 6a69bc8) sent the parameters the node held after the rate loop, which had been restored to the starting values, so rank 0 could adopt untrained parameters whenever the winning rate was not on rank 0. The RP results in the paper were obtained with the corrected version in this repository, with p = 2, 4, 8 and 16.

The two drivers are the ones that were run. To build one of them exactly as it was run, copy it over `src/main.cpp`. `src/main.cpp` itself is an equivalent driver with the same parameters that can also select data sets.

In all three variants the update of the gating vector uses the cached `outputs` of the two children (small random vectors drawn in the constructor) in place of the difference of their output vectors `w0s`. The trees are therefore fitted by this procedure and not by exact gradient descent. The paper describes this in its Section 3.1 and its limitations, and the results in the paper were obtained with it.

The initial weights are drawn with `rand()` and the instance order is produced by `std::shuffle`. Both differ between operating systems and standard libraries. For a given seed the tree is reproducible on the same platform, but not across platforms: the error rates obtained on macOS differ from those obtained on Linux.
