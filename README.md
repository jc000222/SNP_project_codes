# SNP_project_codes
Codes for S&amp;P intern project

# EDARunner: Exploratory Data Analysis Tool for Time Series Data

## Overview

`EDARunner` is a Python class designed to perform exploratory data analysis (EDA) on time series data. It provides various methods to visualize and analyze the data, including histogram, boxplot, autocorrelation, partial autocorrelation, overview, and time series decomposition plots.

## Installation

To install the package from the GitHub repository, use the following command:

```bash
pip install --upgrade git+https://github.com/jc000222/SNP_project_codes.git
```

# Functions
### __init__(self, df, data_name, Show_info=True)
Initialize EDARunner class with the input DataFrame. Make columns to "Timestamp", "Values".

Args:
df (DataFrame): Input DataFrame for analysis.
data_name (str): The name of the dataset.
Show_info (Boolean): Default as True. Show the info of df.

### histogram(self)
Plots a histogram of the values in the DataFrame.

Returns: None
### boxplot(self)
Plots a boxplot of the values in the DataFrame.

Returns: None
### autocorrelation(self)
Plots the autocorrelation and partial autocorrelation functions of the time series.

Returns: None. Displays the autocorrelation and partial autocorrelation plots.
### overview(self, Zoom_in=None)
Plots an overview of the synthetic time series.

Parameters:
Zoom_in: tuple (start_index, end_index) to specify the range of data points to zoom in on. If None, plots the entire series.
Returns: None. Displays the overview plot.
### plot_decomposition(self)
Plots the individual components and the combined series of a time series decomposition.

Returns: None. Displays the decomposition plot.

# Installation and Update Instructions
### Installing the Package

To install your package from a GitHub repository, use the following command:

pip install git+https://github.com/jc000222/SNP_project_codes.git

This command fetches the latest version of the package from the repository and installs it.

### Updating the Package

To update the package to the latest version from the repository, you can use the --upgrade flag:

pip install --upgrade git+https://github.com/jc000222/SNP_project_codes.git

### Restarting the Environment

After installing or updating a package, it's often necessary to restart the Python environment to ensure that the changes take effect. This can be particularly important in environments like Jupyter notebooks. If update is not working, try:
pip uninstall snp
and try install again.


