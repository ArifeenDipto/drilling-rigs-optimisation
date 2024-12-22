# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:23:13 2024

@author: mursh
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Plotter:
    """
    A class for creating various types of plots, including correlation matrices and general-purpose plots.
    """

    def __init__(self, config):
        """
        Initialize the Plotter class.
        """
        self.save_path = config['paths']['figure']
    def plot_correlation_matrix(self, data, title="Correlation Matrix", save_path=True, show=True, **kwargs):
        """
        Plot a correlation matrix using a Seaborn heatmap.

        Parameters:
        - data (pd.DataFrame): DataFrame for which the correlation matrix is calculated.
        - title (str): Title of the heatmap.
        - save_path (str): Path to save the heatmap as an image file (optional).
        - show (bool): Whether to display the plot interactively.
        - kwargs: Additional keyword arguments for Seaborn's heatmap function (e.g., `annot`, `cmap`).
        """
        corr_matrix = data.corr()
        plt.figure(figsize=kwargs.pop("figsize", (10, 8)))
        sns.heatmap(corr_matrix, annot=kwargs.pop("annot", True), cmap=kwargs.pop("cmap", "coolwarm"),
                    fmt=".2f", linewidths=0.5, **kwargs)
        plt.title(title, fontsize=16)

        if save_path:
            plt.savefig(self.save_path, bbox_inches="tight")
            print(f"Correlation matrix heatmap saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def general_plot(self, data, plot_type='line', x=None, y=None, title=None, xlabel=None, ylabel=None,
                     save_path=None, show=True, **kwargs):
        """
        General-purpose plotting function for line, scatter, bar, and histogram plots.

        Parameters:
        - data (pd.DataFrame): Data to visualize.
        - plot_type (str): Type of plot ('line', 'scatter', 'bar', 'hist').
        - x (str or array-like): X-axis data or column name (if data is a DataFrame).
        - y (str or array-like): Y-axis data or column name (if data is a DataFrame).
        - title (str): Title of the plot.
        - xlabel (str): Label for the X-axis.
        - ylabel (str): Label for the Y-axis.
        - save_path (str): Path to save the plot (optional).
        - show (bool): Whether to display the plot interactively.
        - kwargs: Additional keyword arguments for Matplotlib plotting functions.
        """
        plt.figure(figsize=kwargs.pop("figsize", (8, 6)))

        if plot_type == 'line':
            plt.plot(data[x], data[y], **kwargs)
        elif plot_type == 'scatter':
            plt.scatter(data[x], data[y], **kwargs)
        elif plot_type == 'bar':
            plt.bar(data[x], data[y], **kwargs)
        elif plot_type == 'hist':
            plt.hist(data[y], **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        if title:
            plt.title(title, fontsize=14)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_predictions(self, y_true, y_pred, show=True):
        """
        Plot predicted vs. true values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - save_path (str): Path to save the plot (optional).
        - show (bool): Whether to display the plot interactively.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(y_true, label='true')
        plt.plot(y_pred, label='predict')
        #plt.plot(y_true, y_pred, alpha=0.7, color="blue")
        #plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
        plt.legend()
        plt.grid()
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs. Ground Truth")

        
        plt.savefig(self.save_path, bbox_inches="tight")
        print(f"Prediction plot saved")
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_support_vector(self, X_train, feature_1, feature_2, X_support, show = True):
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train[:, feature_1], X_train[:, feature_2], color="blue", label="Training Data")
        plt.scatter(X_support[:, feature_1], X_support[:, feature_2], color="green", edgecolor="black", s=100, label="Support Vectors", marker="o")
        plt.xlabel(f"Feature {feature_1}")
        plt.ylabel(f"Feature {feature_2}")
        plt.title("Support Vectors in Feature Space")
        plt.legend()
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def plot_residuals(self, target, residuals, show = True):
        plt.figure(figsize=(8, 6))
        plt.scatter(target, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        if show:
            plt.show()
        else:
            plt.close()
