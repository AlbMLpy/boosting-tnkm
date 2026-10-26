# Base image with Miniconda
FROM condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Copy conda environment file and create environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Make conda environment active in all commands
SHELL ["conda", "run", "-n", "boosting_env", "/bin/bash", "-c"]

# Copy your project code and data
COPY . .
