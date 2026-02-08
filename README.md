# AE 598 ARP (Advanced Robotic Planning)

This repository has materials for a special-topics course on robot planning in Spring, 2026 at the University of Illinois Urbana-Champaign. Some of these materials are based on similar courses that have been offered by Nicolas Mansard and Ludovic Righetti - thanks to both of them for their support.

## Getting started

These instructions assume you know how to open a terminal and use the command line.

### Get the code

Change your working directory to wherever you want to put your work this semester and clone the repo you're looking at:
```zsh
git clone https://github.com/tbretl/ae598-arp.git
```

### Install conda

I use [miniforge](https://github.com/conda-forge/miniforge) to install `conda`.

I do not suggest you use [anaconda](https://anaconda.org) for this purpose - it causes trouble. If you already have anaconda installed, I suggest you remove it.

It is ok to use [miniconda](https://docs.conda.io/projects/miniconda/) - I used to do this - but if you do, I suggest that you configure your conda environment to use `conda-forge` as the only channel. To do this, you'd run the following commands just after creating and activating an empty envirionment:
```zsh
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```
This configuration is done by default in every environment if you use `miniforge`.

### Create a conda environment

Change your working directory to `ae598-arp` (the repository you just cloned) and run the following command:
```zsh
conda env create -f environment.yml
```
This will create a conda environment that is called `ae598-arp` and that has all the python packages you need.

You may see something like `WARNING: A conda environment already exists` and be asked if you want to `Remove existing environment`. This means you already created an environment called `ae598-arp`. Perhaps something went wrong and you are creating it again — in this case, type `y` to remove the existing environment and proceed to recreate it.

Make sure this process completes successfully. If you see any errors, ask for help.

### Install VS Code

[Visual Studio Code](https://code.visualstudio.com/) is a cross-platform code editor that we recommend for use with this course. Carefully follow *all* of the [instructions for installation and setup](https://code.visualstudio.com/docs/setup/setup-overview) that are particular to your operating system.

To run VS Code, open a terminal, change the working directory to wherever you have your files, and run the command `code .` (i.e., the word “code”, then a space, then a period).

To work with jupyter notebook files (`.ipynb`) in VS Code, you need to install the [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) to VS Code. To do this, open VS Code if you haven’t already, then click on Extensions in the toolbar on the left side of the window.

Install the python extension as follows:

* Type “python” into the search bar.
* Click on **Python** (it should be at the top of the list).
* Click on **Install** in the window that opens.

Install the jupyter extension as follows:

* Type “jupyter” into the search bar.
* Click on **Jupyter** (it should be at the top of the list).
* Click on **Install** in the window that opens.

### Activate your conda environment

To run python on the command line in a terminal, you first need to activate your conda environment:
```zsh
conda activate ae598-arp
```
You should see the prefix on the command line change from `(base)` to `(ae598-arp)`.

To run python in a jupyter notebook with VS Code, you first need to use the [kernel picker](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) to activate the `ae598-arp` environment.