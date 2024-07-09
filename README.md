# Foundation Models for Time Series Research

## Overview

This repository contains research work on foundation models for time series analysis. The goal is to evaluate and compare various foundation models on different time series datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Models](#models)
- [Notebooks](#notebooks)
- [Source](#source)

## Project Structure

For the repository, we use the following [guidelines](https://drivendata.github.io/cookiecutter-data-science/).


```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
```

## Data

The datasets used in this project are stored in the 'data' directory.

## Models

The trained models are stored in the 'models' directory

## Notebooks

Jupiter notebooks for data analysis, experimentation, plotting ...

## Source

'src' directory contains all of the source code