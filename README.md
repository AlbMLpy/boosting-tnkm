<div align="center">

<h1>Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines</h1>

<b> Authors: </b> <a href="https://albmlpy.github.io/">Albert Saiapin</a>, Kim Batselier

[ArXiv]() | [Experiments]()

<p align="center">
  <a href=""><img src="assets/predictions.png?raw=True" alt="Figure 1" width="70%"></a>
</p>

</div>

## 💫 Project Description
In progress ...


## 📈 Datasets
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


## ⚗️ How to Reproduce the Numerical Experiments

0. **Activate** the virtual environment:
    ```shell
    conda activate boosting_env
    ```

1. **Run:**
   ```shell
   python setup_project.py
   ```
   to download all datasets and configure the project directories. 

2. Once the setup script has completed, **run:**
    ```shell
    cd experiments
    ```
    This folder contains two subdirectories, each corresponding to a distinct experiment described in the paper: `base_learners` and `hypers_settings`.

3. `base_learners`
    -  **Run:** `cd base_learners`
    -  **Train**: 
        ```bash
        python experiment.py all
        ```
        Computes evaluation metrics for further comparison. These are stored in `artifacts/training_artifacts`.
    -  **Analyze:** Run `analysis.ipynb` in VS Code using the `boosting_env` environment to generate figures/tables stored in `./artifacts`.

4. `hypers_settings`
    -  **Run:** `cd hypers_settings`
    -  **Train**: 
        ```bash
        python experiment.py all all
        ```
        Computes evaluation metrics for further comparison. These are stored in `artifacts/training_artifacts`.
    -  **Analyze:** Run `analysis.ipynb` in VS Code using the `boosting_env` environment to generate figures/tables stored in `./artifacts`. 


## 📜 Citation

If you find our work helpful, please consider citing the paper:

```bibtex
In Progress!
```
