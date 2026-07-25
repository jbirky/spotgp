# Installation

There are two ways to get set up, depending on what you want to do:

| | What it gives you | Start here if you want to... |
|---|---|---|
| **`spotgp`** | The base package: GP kernels, spot evolution models, and solvers you import into your own scripts and notebooks. | Write your own analysis code, or follow the tutorials in these docs. |
| **`spotgp-project`** | A reproducible analysis workflow built on top of `spotgp`: YAML-driven runs, HDF5 results, DVC pipelines, SLURM/container support, and an interactive Streamlit explorer. | Fit real light curves as a managed, reproducible project, especially on remote machines or cluster environment. |

The project workflow installs the base package for you, so you only need to follow
one of the two install paths below.

<br>

## Set up an environment

Whichever path you take, it is strongly recommended to install into a fresh conda
environment rather than your base environment. `spotgp` pulls in JAX and a full
scientific stack, and the project workflow adds pinned versions of its own — keeping
them isolated avoids breaking other projects and makes the install easy to throw away
and redo:

```bash
conda create -n spotgp python=3.12
conda activate spotgp
```

`spotgp` requires Python 3.10 or newer, and is tested on 3.10, 3.11, and 3.12.
Run all of the install commands below inside the activated environment, and remember
to re-activate it (`conda activate spotgp`) in any new shell — including in SLURM job
scripts, if you are not using the container.

:::{note}
If you do not use conda, the standard library equivalent works just as well:
```bash
python -m venv ~/.venvs/spotgp
source ~/.venvs/spotgp/bin/activate
```
:::

<br>

## Base package: `spotgp`

### From PyPI

```bash
pip install spotgp
```

### From source

```bash
git clone https://github.com/jbirky/spotgp.git
cd spotgp
pip install -e .
```

### Optional extras

Install all optional features (PGM rendering, `jaxopt`, and the spectral
contrast model with `pyphot` + `Korg.jl`) in one shot:

```bash
pip install "spotgp[extras]"
```

Or pick individual extras:

```bash
pip install "spotgp[pgm]"        # daft PGM rendering
pip install "spotgp[jaxopt]"     # jaxopt optimizers
pip install "spotgp[spectral]"   # pyphot bandpasses
pip install "spotgp[korg]"       # pyphot + Korg.jl model atmospheres
```

`Korg.jl` itself is installed from Python after the extras:

```bash
python -c "import juliapkg; juliapkg.add('Korg', 'acafc109-a718-429c-b0e5-afd7f8c7ae46'); juliapkg.resolve()"
```

Alternatively, clone the repo and add it to your Python path:

```bash
git clone https://github.com/jbirky/spotgp.git
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/spotgp"' >> ~/.bashrc
source ~/.bashrc
```

### Check the install

```bash
python -c "import spotgp; print(spotgp.__version__)"
```

<br>

## Project workflow: `spotgp-project`

[`spotgp-project`](https://jessicabirky.com/spotgp-project/) is a template repository
for running reproducible GP analyses of TESS, Kepler, and K2 light curves. Instead of
writing a fit script per star, you describe each run in a YAML config and the pipeline
handles the rest: MAP optimization or MCMC sampling, self-contained HDF5 results with
the config embedded, DVC-tracked outputs, and optional Weights & Biases / MLflow
experiment tracking.

### Local install

```bash
git clone https://github.com/jbirky/spotgp-project.git
cd spotgp-project
pip install -r requirements.txt
```

This pulls in `spotgp` (with JAX), the `dynesty` fitting backend, the Streamlit
explorer, and the rest of the pipeline dependencies — so you do not need to install
the base package separately.

### On a cluster

For HPC systems, pull the pre-built Apptainer container rather than installing into
your home directory:

```bash
mkdir -p ~/containers
apptainer pull ~/containers/spotgp.sif docker://ghcr.io/<org>/spotgp:latest
```

Jobs are submitted with the bundled SLURM scripts, and `SPOTGP_SIF` overrides the
container path:

```bash
SPOTGP_SIF=/path/to/spotgp.sif sbatch scripts/run_fit.slurm configs/my_star.yaml
```

### Running a fit

The `Makefile` wraps the common entry points:

```bash
make run CONFIG=configs/example.yaml             # run locally
make run-container CONFIG=configs/example.yaml   # run via apptainer
make submit CONFIG=configs/example.yaml          # submit to SLURM
make validate CONFIG=configs/example.yaml        # check a config without fitting
make app                                         # launch the Streamlit explorer
```

:::{tip}
See the [spotgp-project installation guide](https://jessicabirky.com/spotgp-project/installation/)
for the full setup — optional tracking backends (`blackjax`, `wandb`, `mlflow`), the
project directory layout, and how to run the test suite.
:::
