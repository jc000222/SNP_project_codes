import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

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


    def histogram(self):
        '''
        Plots a histogram of the values in the DataFrame.
        '''
        plt.figure(figsize=(5, 3))
        sns.histplot(self.df['Values'], bins=50, kde=True)
        plt.title('Histogram of Values')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()
        
    def boxplot(self):
        '''
        Plots a boxplot of the values in the DataFrame.
        '''
        plt.figure(figsize=(5, 3))
        sns.boxplot(self.df['Values'])
        plt.show()

    def autocorrelation(self):
        '''
        Plots the autocorrelation and partial autocorrelation functions of the time series.

        Parameters:
        None
        
        Returns:
        None. Displays the autocorrelation and partial autocorrelation plots.
        '''
        # Plot the autocorrelation
        fig, ax = plt.subplots(figsize=(15, 3))
        pd.plotting.autocorrelation_plot(self.df['Values'], ax=ax)
        ax.set_title(f'Autocorrelation Plot - {self.data_name}')
        ax.set_ylim(-1, 1)  # Set y-axis limits
        ax.grid(True)
        plt.show()

        # Plot the partial autocorrelation
        fig, ax = plt.subplots(figsize=(15, 3))  # Ensure both plots have the same size
        plot_acf(self.df['Values'], lags=40, ax=ax)
        ax.set_title(f'Partial Autocorrelation Plot - {self.data_name}')
        ax.grid(True)
        plt.show()
    
    def overview(self, Zoom_in=None):
        '''
        Plots an overview of the synthetic time series.

        Parameters:
        - Zoom_in: tuple (start_index, end_index) to specify the range of data points to zoom in on. If None, plots the entire series.
            
        Returns:
        None. Displays the overview plot.
        '''
        # Plot the synthetic time series
        plt.figure(figsize=(10, 6))
        
        if Zoom_in:
            plt.plot(self.df['Timestamp'][Zoom_in[0]:Zoom_in[1]], self.df['Values'][Zoom_in[0]:Zoom_in[1]], label="Synthetic Time Series")
        else:
            plt.plot(self.df['Timestamp'], self.df['Values'], label="Synthetic Time Series")
        
        plt.xlabel("Timestamp")
        plt.ylabel("Values")
        plt.title(self.data_name)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_decomposition(self):
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

        plt.show()
