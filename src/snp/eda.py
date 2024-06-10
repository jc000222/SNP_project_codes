"""
eda.py contains a EDARunner class to perform EDA.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import warnings


class EDARunner:
    """
    Perform EDA for different datafram from data generation step.
    """
    
    def __init__(self, df, data_name, Show_info=True):
        """
        Initialize EDARunner class with the input DataFrame. Make columns to "Timestamp", "Values".

        Args:
        - df (DataFrame): Input DataFrame for analysis.
        - data_name (str): The name of the dataset.
        - Show_info (Boolean): Default as True. Show the info of df.
        """
        df.columns = ["Timestamp", "Values"]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        self.df = df
        self.data_name = data_name
        self.default_file_path = f'/EDA_Result/{data_name}'
        self.colors = [
        '#1f77b4',  # Blue -- For the main data
        '#ff7f0e',  # Orange -- For seasonality
        '#2ca02c',  # Green -- For linearity
        '#d62728'  # Red -- For residual
        ]
        if Show_info:
            display(df.info())
            display(df.describe())
            display(df.head())
            df.sort_values(by="Timestamp", inplace=True)
            # Infer the frequency of the time series
            frequency = pd.infer_freq(df["Timestamp"])

            # Calculate the time period
            time_period = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]

            print("Time Period:", time_period)
            print("Frequency:", frequency)
        # Suppress specific warning from Matplotlib
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")


    def histogram(self, save_plot=False, file_path=None, color=None):
        '''
        Plots a histogram of the values in the DataFrame.
        '''
        plt.figure(figsize=(5, 3))
        sns.histplot(self.df['Values'], bins=50, kde=True, color=color)
        plt.title('Histogram of Values')
        plt.xlabel('Values')
        plt.ylabel('Frequency')

        if save_plot:
            if file_path == None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            file = file_path + f"/histogram_{self.data_name}.png"
            plt.savefig(file)
            print(f'Plot saved to {file}')
        plt.show()
        
    def boxplot(self, save_plot=False, file_path=None, color=None):
        '''
        Plots a boxplot of the values in the DataFrame.
        '''
        plt.figure(figsize=(5, 3))
        sns.boxplot(self.df['Values'],color=color)

        if save_plot:
            if file_path == None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            file = file_path + f"/boxplot_{self.data_name}.png"
            plt.savefig(file)
            print(f'Plot saved as {file}')
        plt.show()

    def autocorrelation(self, save_plot=False, file_path=None, Partial_plot=False, color=None):
        '''
        Plots the autocorrelation and partial autocorrelation functions of the time series.

        Parameters:
        None
        
        Returns:
        None. Displays the autocorrelation and partial autocorrelation plots.
        '''
        if Partial_plot:
            # Plot the partial autocorrelation
            fig, ax = plt.subplots(figsize=(15, 3))  # Ensure both plots have the same size
            plot_acf(self.df['Values'], lags=40, ax=ax,color=color)
            ax.set_title(f'Partial Autocorrelation Plot - {self.data_name}.png')
            ax.grid(True)
        else:
            # Plot the autocorrelation
            fig, ax = plt.subplots(figsize=(15, 3))
            pd.plotting.autocorrelation_plot(self.df['Values'], ax=ax,color=color)
            ax.set_title(f'Autocorrelation Plot - {self.data_name}.png')
            ax.set_ylim(-1, 1)  # Set y-axis limits
            ax.grid(True)
        
        if save_plot:
            if file_path == None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            file = file_path + f"/autocorrelation_{self.data_name}.png"
            plt.savefig(file)
            print(f'Plot saved as {file}')
        plt.show()
    

    def overview(self, save_plot=False, Zoom_in=None, file_path=None, color=None):
        '''
        Plots an overview of the synthetic time series.

        Parameters:
        - save_plot: bool, if True, saves the plot to the specified file path.
        - Zoom_in: tuple (start_index, end_index) to specify the range of data points to zoom in on. If None, plots the entire series.
        - file_path: str, the directory path where the plot should be saved.
        - color: str, the color of the plot line.

        Returns:
        None. Displays the overview plot.
        '''
        # Plot the synthetic time series
        plt.figure(figsize=(10, 6))
        
        if Zoom_in:
            plt.plot(self.df['Timestamp'][Zoom_in[0]:Zoom_in[1]], self.df['Values'][Zoom_in[0]:Zoom_in[1]], label="Synthetic Time Series", color=color)
        else:
            plt.plot(self.df['Timestamp'], self.df['Values'], label="Synthetic Time Series", color=color)
        
        plt.title(self.data_name)
        plt.grid(True)
        
        # Define date formatting
        date_format = mdates.DateFormatter('%m/%d/%y')
        
        # Set x-axis date format
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        if save_plot:
            if file_path is None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            if Zoom_in:
                file = os.path.join(file_path, f"overview_{self.data_name}_{Zoom_in[0]}_{Zoom_in[1]}.png")
            else:
                file = os.path.join(file_path, f"overview_{self.data_name}.png")
            plt.savefig(file)
            print(f'Plot saved as {file}')
        plt.show()


    def plot_decomposition(self, save_plot=False, file_path=None):
        '''
        Plots the individual components and the combined series of a time series decomposition.

        Parameters:
        None
        
        Returns:
        None. Displays the decomposition plot.
        '''
        # Decompose the time series data into its components: observed, trend, seasonal, and residual
        decomposition = seasonal_decompose(self.df["Values"], model='additive', period=365)

        # Create a figure with 4 subplots, one for each component
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

        # Set the title of the entire figure
        fig.suptitle(self.data_name, fontsize=16)
        
        # Plot the observed data
        decomposition.observed.plot(ax=ax1, color=self.colors[0])
        ax1.set_ylabel('Observed')

        # Plot the seasonal component
        decomposition.seasonal.plot(ax=ax3, color=self.colors[1])
        ax3.set_ylabel('Seasonal')
        
        # Plot the trend component
        decomposition.trend.plot(ax=ax2, color=self.colors[2])
        ax2.set_ylabel('Trend')
        
        # Plot the residual component
        decomposition.resid.plot(ax=ax4, color=self.colors[3])
        ax4.set_ylabel('Residual')
        
        if save_plot:
            if file_path == None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            file = file_path + f"/decomposition_{self.data_name}.png"
            plt.savefig(file)
            print(f'Plot saved as {file}')
        plt.show()




class MultiEDARunner:
    def __init__(self, df,data_name, Show_info=True, colormap=None):
        self.df=df
    
        self.data_name = data_name
        self.default_file_path = f'/EDA_Result/{data_name}'
        # List of colormaps to randomly select from
        colormaps = [
            plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Purples, plt.cm.Oranges, 
            plt.cm.BuPu, plt.cm.YlGnBu,
            plt.cm.YlOrBr, plt.cm.OrRd, plt.cm.PuBu, plt.cm.GnBu]

        # Randomly select one colormap for this instance
        self.selected_colormap = colormap if colormap else np.random.choice(colormaps)
        

        if Show_info:
            print("Columns:",df.columns)
            display(df.head())
        # Suppress specific warning from Matplotlib
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    def pacf_threshold(self, nlags=40, threshold=0.5, save_plot=False, file_path=None):
        """
        Plots the lag at which the PACF drops below a specified threshold for each column in the DataFrame.

        Parameters:
        nlags (int): The number of lags to calculate the PACF for. Default is 40.
        threshold (float): The threshold for PACF. Default is 0.5.
        color (str): The color of the bars in the plot. Default is None.

        Returns:
        None. Displays the bar plot.
        """
        lag_below_threshold = {}

        # Calculate the PACF for each column except the Timestamp
        for column in self.df.columns[1:]:
            pacf_values = pacf(self.df[column], nlags=nlags)
            # Find the first lag where PACF drops below the threshold
            lag = next((i for i, value in enumerate(pacf_values) if abs(value) < threshold), None)
            lag_below_threshold[column] = lag

        # Extract the relevant parts of the column names
        column_labels = [col[3:].replace('p', '.') for col in lag_below_threshold.keys()]

        # Plotting
        plt.figure(figsize=(18, 3))
        plt.bar(column_labels, lag_below_threshold.values(), color=self.selected_colormap(0.8))
        plt.ylabel(f'Lag where PACF drops below {threshold}')
        plt.title(f'Lag at which PACF drops below {threshold} for each column')
        plt.xticks(rotation=90)
        plt.grid(True)
        if save_plot:
            if file_path == None:
                file_path = self.default_file_path
            os.makedirs(file_path, exist_ok=True)
            file = file_path + f"/pacf_{self.data_name}.png"
            plt.savefig(file)
            print(f'Plot saved as {file}')

        plt.show()

    def visualize_variance_std_dev(self):
        """
        Computes, prints, and visualizes the variance and standard deviation for each column in the DataFrame.

        Parameters:
        None

        Returns:
        None. Displays the bar plots.
        """
        names = []
        variances = []
        std_devs = []

        # Collecting data
        for column in self.df.columns[1:]:
            variance = np.var(self.df[column])
            std_dev = np.std(self.df[column])
            names.append(column[3:].replace('p', '.'))
            variances.append(variance)
            std_devs.append(std_dev)
        # Visualizing the data
        x = np.arange(len(names))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(18, 5))
        rects1 = ax.bar(x - width/2, variances, width, label='Variance')
        rects2 = ax.bar(x + width/2, std_devs, width, label='Standard Deviation')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Values')
        ax.set_title('Variance and Standard Deviation by Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.legend()

        # Attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 4)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()



    def overview(self, save_plot=False, Zoom_in=None, file_path=None, columns=None):
        '''
        Plots an overview of the synthetic time series for each column in the DataFrame.

        Parameters:
        - Zoom_in: tuple (start_index, end_index) to specify the range of data points to zoom in on. If None, plots the entire series.
        - save_plot: bool, if True, saves the plot to the specified file path.
        - file_path: str, the directory path where the plot should be saved.
        - columns: list of str, columns to be plotted. If None, all columns except Timestamp are plotted.

        Returns:
        None. Displays or saves the overview plot.
        '''
        if columns is None:
            columns = self.df.columns[1:]  # Exclude Timestamp column
        else:
            columns = pd.Index(columns)
        num_columns = len(columns)
        
        # Determine the number of rows needed, 4 subplots per row
        num_rows = (num_columns + 3) // 4

        fig, axs = plt.subplots(num_rows, 4, figsize=(18, 3 * num_rows), sharey=True, constrained_layout=True)
        axs = axs.flatten()  # Flatten in case we have a single row to make indexing easier
        fig.suptitle(self.data_name)

        if file_path is None:
            file_path = self.default_file_path
        os.makedirs(file_path, exist_ok=True)

        # Using the selected colormap for progressively darker colors
        colormap = self.selected_colormap
        colors = [colormap(i / (num_columns+4)) for i in range(4,num_columns+4)]
        for i, column in enumerate(columns):
            plot_color = colors[i]
            if Zoom_in:
                fig.suptitle(self.data_name + f" | Zoomed {Zoom_in}")
                axs[i].plot(self.df['Timestamp'][Zoom_in[0]:Zoom_in[1]], self.df[column][Zoom_in[0]:Zoom_in[1]], label=column, color=plot_color)
            else:
                axs[i].plot(self.df['Timestamp'], self.df[column], label=column, color=plot_color)
            
            axs[i].set_title("std: " + column[3:].replace('p', '.'))
            axs[i].grid(True)

            # Define date formatting
            date_format = mdates.DateFormatter('%m/%d/%y')
            axs[i].xaxis.set_major_formatter(date_format)
            axs[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_plot:
            if Zoom_in:
                file = os.path.join(file_path, f"overview_{self.data_name}_{Zoom_in[0]}_{Zoom_in[1]}.png")
            else:
                file = os.path.join(file_path, f"overview_{self.data_name}.png")
            plt.savefig(file)
            print(f'Plot saved as {file}')

        plt.show()