<div align="center">

<h1>Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines</h1>

<b> Authors: </b> <a href="https://albmlpy.github.io/">Albert Saiapin</a>, Kim Batselier

[ArXiv]() | [Experiments](#-how-to-reproduce-the-numerical-experiments)

<p align="center">
  <a href=""><img src="assets/diagram.png?raw=True" alt="Figure 1" width="80%"></a>
</p>

**Figure 1:** Comparison of the training procedures for CP tensor network kernel machines (CPKMs) using gradient boosting (GB) and rank selection via validation (CV). (a) GB-CPKM incrementally builds an ensemble by training fixed-rank $\hat{R}$ base learners and concatenating their factor matrices along the rank dimension to represent the combined model. (b) In contrast, CV-CPKM independently trains candidate models with different CP ranks and selects the rank that achieves the lowest validation error.

</div>

## 🌟 Project Description
This repository provides the implementation of **GB-CPKM**, a gradient boosting framework for tensor network kernel machines (TNKMs) based on the canonical polyadic (CP) decomposition. The proposed approach incrementally builds a predictive model by combining fixed-rank base learners, providing an alternative to conventional rank-selection methods that train multiple models with different CP ranks.

The repository contains the code and experimental configurations used to evaluate GB-CPKM on regression benchmarks of varying sizes and dimensionalities. It supports reproducing the experiments reported in our paper, including comparisons with a validation-based rank-selection baseline in terms of predictive performance, training time, and peak memory consumption.


## 📉 Datasets
For ablation studies and real-data experiments, we use the following 4 UCI regression datasets (Dua and Graff, 2017), Airline (Hensman et al., 2013), Precipitation (Dong et al., 2017) and Kin8nm (10.5281/zenodo.14645865) datasets:

|  | Energy | Concrete | Kin8nm | Naval | Precipitation | HouseElectric | Airline |
| ------------- | ------------- | ------------- | ------------- | ------------- | -------------| -------------| -------------|
| **N** | 768 | 1030 | 8192 | 11934 | 628474 | 2049280 | 5929413 |
| **D** | 8 | 8 | 8 | 16 | 3 | 6 | 8 |

where **N** represents the number of samples and **D** denotes the data dimensionality.


## ⚙️ Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, **run** the following command (while in the root of the repository):
```shell
conda env create -f environment.yml
```
This will create a new environment named `boosting_env` with all required packages already installed. You can **install** additional packages by running:
```shell
conda install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`


## 🐳 (Optional) Running Experiments with Docker

Instead of setting up a Conda environment manually, you can run the entire experiment inside a Docker container. This ensures full reproducibility and requires only `Docker to be installed` on your system. 

ℹ️ Note: Depending on your Docker installation, you may need to prefix
all `docker` commands in this guide with `sudo`. 

1. From the project root (where the `Dockerfile` is located) **build** the Docker image:
    ```shell
    docker build -t gb-tnkm-project .
    ```

2. **Run** the container interactively:
    ```shell
    docker run -it -v $(pwd)/experiments:/app/experiments --name gb-tnkm gb-tnkm-project
    ```
3. **You are all set to reproduce the [Numerical Experiments](#-how-to-reproduce-the-numerical-experiments)!** 🤗

4. **Re-enter** the same container:
    ```shell
    docker start -ai gb-tnkm
    ```

5. **Cleaning up** (optional):
    
    1. Remove the container:
        ```shell
        docker rm gb-tnkm
        ```
    2. Remove the image:
        ```shell
        docker rmi gb-tnkm-project
        ```


## ⚗️ How to Reproduce the Numerical Experiments

Run the following commands from the repository root unless noted otherwise.

1. **Activate the environment and prepare the project:**
    ```shell
    conda activate boosting_env
    python setup_project.py
    ```
    This downloads the datasets and configures the project directories.

2. **Run the base-learners experiment:**
    ```shell
    cd experiments/base_learners
    python experiment.py all
    ```
    This runs all base-learners experiments and saves the results under `./artifacts/training_artifacts` directory.

    To generate the resulting figures, run:
    ```shell
    python analysis.py
    ```

3. **Run the hyperparameter-settings experiment:**

    From `base_learners`, move to the other experiment directory:
    ```shell
    cd ../hypers_settings
    python experiment.py all all
    ```
    This runs all hyperparameter-settings experiments and saves the results under `./artifacts/training_artifacts` directory.

    To generate the resulting figures and tables, run:
    ```shell
    python analysis.py
    ```


## 📜 Citation

If you find this repository useful in your work, please cite the paper:

```bibtex
In Progress!
```
