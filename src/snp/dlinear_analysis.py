import ast
import os
import zipfile
import pandas as pd
import subprocess
import re
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import platform
import seaborn as sns
from data_provider.data_factory import data_dict
from data_provider.data_loader import Dataset_Custom

class DataPipeline:
    def __init__(self,
                 model_name='DLinear',
                 seq_len=336,
                 pred_lens=[24, 48, 96, 192, 336, 720],
                 batch_size=32,
                 learning_rate=0.005,
                 feature='S'):
        """
        This class defines a data processing pipeline that handles data extraction,
        preprocessing, training, and analysis for time series models. It supports
        various functionalities from handling zip files to generating and analyzing
        reports.

        Note: This class implementation was partly assisted by ChatGPT with modifications
        and additional documentation.

        Parameters:
            model_name (str): The name of the model to use for training.
            seq_len (int): Sequence length for the model.
            pred_lens (list): List of prediction lengths.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for training.
            feature (str): Feature type for the model, "S" for univariate, "M" for multivariate.
        """
        self.model_name = model_name
        self.seq_len = seq_len
        self.pred_lens = pred_lens
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.feature = feature
        self.is_windows = platform.system().lower() == 'windows'

    def extract_zip(self, file_path, extract_to='dataset'):
        """
        Extracts the contents of a zip file to the specified directory.

        Parameters:
            file_path (str): Path to the zip file.
            extract_to (str): Directory to extract files to.
        """
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {file_path} to {extract_to}")

    def update_python_dict_file(self, file_path):
        """
        Append a new key-value pair to a dictionary in a Python file if it does not already exist.
        This uses a templating approach to ensure class references are added correctly.

        Args:
        file_path (str): Path to the Python file where the dictionary is defined.
        """
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Prepare the new dictionary content
        dict_entries = []
        for k, v in data_dict.items():
            # Assuming v is a class or a class instance, get its name for proper referencing
            class_name = v.__name__ if hasattr(v, '__name__') else repr(v)
            dict_entries.append(f"'{k}': {class_name}")

        # Join all entries to form the dictionary string
        dict_content = ',\n    '.join(dict_entries)

        # Replace or append the updated dictionary in the file
        with open(file_path, 'w') as file:
            dictionary_found = False
            in_dict_definition = False
            for line in lines:
                if re.match(r'^\s*data_dict\s*=\s*{', line):
                    # Write the updated dictionary
                    file.write(f"data_dict = {{\n    {dict_content}\n}}\n")
                    dictionary_found = True
                    in_dict_definition = True
                    continue
                elif in_dict_definition:
                    # Check for the closing brace of the dictionary
                    if '}' in line:
                        in_dict_definition = False
                    continue  
                file.write(line)
            # If the dictionary definition was not found, append it at the end
            if not dictionary_found:
                file.write(f"data_dict = {{\n    {dict_content}\n}}\n")

    def create_sh_file(self,
                       folder_name,
                       dataset_name,
                       model_name=None,
                       seq_len=None,
                       pred_lens=None,
                       batch_size=None,
                       learning_rate=None,
                       feature=None,
                       use_gpu=True,
                       gpu=0,
                       devices='0,1,2,3'):
        """
        Creates a shell script for running the training process on a given dataset.

        Parameters:
            folder_name (str): Name of the base folder for datasets.
            dataset_name (str): Name of the dataset file.
            model_name (str): The name of the model to use for training. If None, uses the instance's model_name.
            seq_len (int): Sequence length for the model. If None, uses the instance's seq_len.
            pred_lens (list): List of prediction lengths. If None, uses the instance's pred_lens.
            batch_size (int): Batch size for training. If None, uses the instance's batch_size.
            learning_rate (float): Learning rate for training. If None, uses the instance's learning_rate.
            feature (str): Feature type for the model, "S" for univariate, "M" for multivariate. If None, uses the instance's feature.

        Returns:
            str: Path to the created shell script.
        """
        data_dict[dataset_name.split('.')[0]] = Dataset_Custom
        print(data_dict.keys(), data_dict.values())
        print(os.getcwd())
        file_path = './data_provider/data_factory.py'
        self.update_python_dict_file(file_path)
        current_directory = os.getcwd()
        scripts_path = os.path.join(current_directory, 'scripts')
        if not os.path.exists(scripts_path):
            raise FileNotFoundError(f"The 'scripts' directory does not exist in the current working directory: {current_directory}")

        model_name = model_name or self.model_name
        seq_len = seq_len or self.seq_len
        pred_lens = pred_lens or self.pred_lens
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        feature = feature or self.feature

        print(f"Inside create_sh_file: Dataset Name - {dataset_name}")
        dataset_name = dataset_name.replace('.csv', '')
        sh_content = f"""
        if [ ! -d "./logs" ]; then
            mkdir ./logs
        fi

        if [ ! -d "./logs/LongForecasting" ]; then
            mkdir ./logs/LongForecasting
        fi

        if [ ! -d "./logs/LongForecasting/{feature}" ]; then
            mkdir ./logs/LongForecasting/{feature}
        fi

        model_name={model_name}

        """
        for pred_len in pred_lens:
            sh_content += f"""
            python -u run_longExp.py \\
              --is_training 1 \\
              --root_path ./dataset/ \\
              --data_path {dataset_name}.csv \\
              --model_id {dataset_name}_{seq_len}_{pred_len} \\
              --model $model_name \\
              --data {dataset_name} \\
              --seq_len {seq_len} \\
              --pred_len {pred_len} \\
              --enc_in 1 \\
              --des 'Exp' \\
              --itr 1 \\
              --batch_size {batch_size} \\
              --feature {feature} \\
              --learning_rate {learning_rate} \\
              --use_gpu True \\
              --gpu {gpu} \\
              --devices '{devices}' > logs/LongForecasting/$model_name_f'{feature}_{dataset_name}_{seq_len}_{pred_len}.log'
            """
        
        sh_file_path = os.path.join(scripts_path, f'EXP-LongForecasting/Linear/{feature}/run_{dataset_name}.sh')
        os.makedirs(os.path.dirname(sh_file_path), exist_ok=True)
        with open(sh_file_path, 'w') as file:
            file.write(sh_content)
        return sh_file_path

    def run_sh_file(self, sh_file_path):
        """
        Runs the specified shell script.

        Parameters:
            sh_file_path (str): Path to the shell script.
        """
        try:
            if self.is_windows:
                subprocess.run(["cmd", "/c", sh_file_path], check=True)
            else:
                subprocess.run(["bash", sh_file_path], check=True)
            print(f"Successfully ran the script: {sh_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running the script: {sh_file_path}")
            print(e)

    def extract_metrics_and_params(self, log_file_path):
        """
        Extracts metrics and parameters from a log file.

        Parameters:
            log_file_path (str): Path to the log file.

        Returns:
            tuple: Extracted metrics and parameters (mae, mse, test_size, seq_len, pred_len).
        """
        mae, mse, test_size, seq_len, pred_len = None, None, None, None, None
        with open(log_file_path, 'r') as file:
            for line in file:
                if "mae:" in line:
                    mae = float(line.split("mae:")[1].strip())
                if "mse:" in line:
                    mse = float(line.split("mse:")[1].split(",")[0].strip())
                if "test" in line and "test " in line:
                    test_size = int(line.split()[1])
                if "seq_len=" in line:
                    seq_len = int(line.split("seq_len=")[1].split(",")[0].strip())
                if "pred_len=" in line:
                    pred_len = int(line.split("pred_len=")[1].split(",")[0].strip())
        return mae, mse, test_size, seq_len, pred_len

    def preprocess_and_train(self,
                             folder_name,
                             dataset_dir='dataset',
                             report_file='report.csv'):
        """
            Preprocesses datasets, runs training scripts, and collects performance metrics.

            Parameters:
                folder_name (str): Folder Name for the unzipped folder
                dataset_dir (str): Directory containing datasets.
                report_file (str): File to save the report.
        """
        print(f"Folder Name: {folder_name}")
        report = pd.DataFrame(columns=['model', 'dataset_type', 'test_mse', 'test_mae', 'seq_len', 'pred_len'])

        for root, _, files in tqdm(os.walk(f'{dataset_dir}')):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                dataset_path = os.path.join(root, file)
                print(f"Processing file: {dataset_path}")
                df = pd.read_csv(dataset_path)

                # Debugging output
                print(f"Initial columns: {df.columns}")

                # Clean column names
                df.columns = df.columns.str.strip()
                print(f"Cleaned columns: {df.columns}")
                df = df.rename(columns={
                  'Value': 'OT',
                  'Timestamp': 'date'
                })
                if 'date' not in df.columns:
                    raise ValueError(f"Multivariate data must contain a 'date' column. Columns available in {file}: {list(df.columns)}")
                
                df['date'] = pd.to_datetime(df['date'])  # Convert 'date' to datetime

                if self.feature == 'S':
                    # Check if 'OT' column is present
                    if 'OT' not in df.columns:
                        raise ValueError(f"'OT' column not found in the dataset {file}. Columns available: {list(df.columns)}")
                    df = df[['date', 'OT']]  # Keep only 'date' and 'OT' columns
                    df.columns = ['date', 'OT']
                    df['OT'] = StandardScaler().fit_transform(df['OT'].values.reshape(-1, 1))
                else:
                    cols_to_scale = df.columns.difference(['date'])
                    df[cols_to_scale] = StandardScaler().fit_transform(df[cols_to_scale])

                df.to_csv(dataset_path, index=False)
                
                sh_file_path = self.create_sh_file(folder_name,file)
                self.run_sh_file(sh_file_path)

                dataset_name = file.replace('.csv', '')
                for log_file in os.listdir(f'logs/LongForecasting/{self.feature}/'):
                    if dataset_name in log_file:
                        mae, mse, test_size, seq_len, pred_len = self.extract_metrics_and_params(os.path.join(f'logs/LongForecasting/{self.feature}/', log_file))
                        new_data = pd.DataFrame([{'model': self.model_name, 'dataset_type': dataset_name, 'test_mse': mse, 'test_mae': mae, 'seq_len': seq_len, 'pred_len': pred_len}])
                        report = pd.concat([report, new_data], ignore_index=True)
        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)
        report.to_csv("./reports/" + report_file, index=False)
        print(f"Report saved to {report_file}")
    
    def load_reports(self, reports_directory):
        combined_reports = pd.DataFrame()
        for file_name in os.listdir(reports_directory):
            if file_name.endswith('.csv'):
                file_path = os.path.join(reports_directory, file_name)
                try:
                    df = pd.read_csv(file_path)
                    print(df.head())
                    df['source_file'] = file_name  # Add the file name as a column
                    if combined_reports.empty:
                        combined_reports = df
                    else:
                        combined_reports = pd.concat([combined_reports, df], ignore_index=True)
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
        return combined_reports

    def analyze_reports(self, reports_df):
        # Example analysis: summarizing key metrics
        summary = reports_df.describe(include='all')  # Getting a statistical summary
        return summary

    def plot_metrics(self, df):
        # Set a color palette that offers better contrast
        palette = sns.color_palette("viridis", n_colors=df['pred_len'].nunique())

        # Iterate through each unique file in the DataFrame
        for file_name in df['source_file'].unique():
            subset = df[df['source_file'] == file_name]
          
            # Adjust the figure size for better layout
            plt.figure(figsize=(16, 12))
            plt.suptitle(f'Data Analysis for {file_name}', fontsize=16)

            # First plot: Test MSE with improved color contrast
            plt.subplot(2, 2, 1)
            sns.barplot(data=subset, x='seq_len', y='test_mse', hue='pred_len', palette=palette)
            plt.title('Test MSE by Sequence Length and Prediction Length')
            plt.legend(title='Prediction Length')
          
            # Second plot: Test MAE with annotations
            plt.subplot(2, 2, 2)
            bar_plot = sns.barplot(data=subset, x='seq_len', y='test_mae', hue='pred_len', palette=palette)
            plt.title('Test MAE by Sequence Length and Prediction Length')
            # Adding annotations to each bar
            for p in bar_plot.patches:
                bar_plot.annotate(format(p.get_height(), '.2f'), 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', 
                                    xytext=(0, 9), 
                                    textcoords='offset points')

            # Third plot: Correlation matrix with better clarity
            plt.subplot(2, 2, 3)
            corr = subset[['test_mse', 'test_mae', 'seq_len', 'pred_len']].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')

            # Fourth plot: Distribution of Test MAE using distinct styles for histograms and KDE
            plt.subplot(2, 2, 4)
            sns.histplot(subset['test_mae'], kde=True, color='blue', edgecolor='black', linewidth=1.2)
            plt.title('Distribution of Test MAE')
            plt.xlabel('Test MAE')
            plt.ylabel('Density')
            plt.grid(True)

            # Save each figure
            plots_directory = '/content/LTSF-Linear/reports/generated_plots'
            os.makedirs(plots_directory, exist_ok=True)
            plot_file_path = os.path.join(plots_directory, f'plots_for_{file_name.replace(".csv", "")}.png')
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
            plt.savefig(plot_file_path)
            plt.close()   

    def perform_analysis(self, reports_directory='reports'):
        reports_df = self.load_reports(reports_directory)
        reports_df = self.clean_data(reports_df)
        analysis_results = self.analyze_reports(reports_df)
        self.plot_metrics(reports_df)  # This could be commented out if no plots are needed
        return analysis_results
    
    def clean_data(self, df):
        # Assuming specific columns to convert to numeric, excluding 'source_file'
        numeric_cols = df.columns.difference(['model', 'dataset_type', 'source_file'])
        print(f'numeric_cols: {numeric_cols}')
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df.dropna()

    def main(self, zip_file_path):
        """
        Main function to handle the overall workflow.

        Parameters:
            zip_file_path (str): Path to the zip file containing datasets.
        """
        self.extract_zip(zip_file_path)
        folder_name = zip_file_path.split('.')[0]
        self.preprocess_and_train(folder_name)

if __name__ == '__main__':
    zip_file_path = 'Dataset_Task1.zip'  # Replace with the path to your zip file
    file_name = zip_file_path.split('.')[0]
    pipeline = DataPipeline(feature='S')
    pipeline.main(zip_file_path)

    # Uncomment below lines for analysis of reports
    # Perform Analysis on Reports
    # results = pipeline.perform_analysis('reports')
    # print(results)
