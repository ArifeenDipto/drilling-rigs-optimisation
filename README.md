
# Machine Learning Pipeline CLI Tool

This project provides a **Command-Line Interface (CLI)** tool for performing data processing, visualization, feature selection, model training, and evaluation. It supports a flexible YAML configuration system to streamline workflows for various machine learning models.

---

## **Features**

### **1. Tasks**
- **Read Data**: Load raw or processed datasets.
- **Plot**: Generate visualizations such as correlation matrices, prediction plots, residual plots, and support vector analysis.
- **Select Features**: Perform feature selection based on a correlation threshold or other criteria.
- **Train Models**: Train machine learning models such as SVR, PLS, and PCR with specified parameters.
- **Test Models**: Evaluate trained models on test datasets.

### **2. Models Supported**
- **Support Vector Regression (SVR)**: Includes kernel options like linear, polynomial, and RBF.
- **Partial Least Squares (PLS)**: For dimensionality reduction and regression.
- **Principal Component Regression (PCR)**: Combines PCA and linear regression for feature reduction and prediction.

### **3. Plot Types**
- **Correlation Matrix**: Visualize feature correlations.
- **Predictions vs. Ground Truth**: Compare predicted and actual values.
- **Residual Plot**: Analyze model errors.
- **Support Vectors**: Analyze SVR support vectors.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Install Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Running the CLI Tool**
```bash
python main.py --task <task_name> [options]
```

### **2. Command-Line Arguments**

#### **General Arguments**
- `--config`: Path to the YAML configuration file. Defaults to the preconfigured file.
- `--task`: Task to perform. Choices:
  - `read`
  - `plot`
  - `select_features`
  - `train`
  - `test`

#### **Task-Specific Arguments**

| Task            | Arguments                                                                                              |
|-----------------|-------------------------------------------------------------------------------------------------------|
| **read**        | `--data_type`: Type of data to read (`raw` or `processed`).                                            |
| **plot**        | `--plot_type`: Type of plot to generate (`correlation`, `predictions`, `residuals`, `s_vectors`).      |
| **train**       | `--model`: Model to train (`svr`, `pls`, `pcr`).<br>`--kernel`: Kernel type for SVR.<br>`--n_comp`: Number of components for PLS/PCR. |
| **test**        | `--model`: Model to evaluate.                                                                          |

---

## **Examples**

### **1. Read Data**
#### Load Raw Data
```bash
python main.py --task read --data_type raw
```
#### Load Processed Data
```bash
python main.py --task read --data_type processed
```

### **2. Generate Plots**
#### Correlation Matrix
```bash
python main.py --task plot --plot_type correlation --data_type raw
```
#### Predictions Plot
```bash
python main.py --task plot --plot_type predictions --model svr
```
#### Residuals Plot
```bash
python main.py --task plot --plot_type residuals --model svr
```
#### Support Vectors Plot
```bash
python main.py --task plot --plot_type s_vectors --model svr
```

### **3. Train Models**
#### Train SVR with RBF Kernel
```bash
python main.py --task train --model svr --kernel rbf --n_comp 3
```
#### Train PLS
```bash
python main.py --task train --model pls --n_comp 5
```

### **4. Test Models**
#### Evaluate SVR
```bash
python main.py --task test --model svr
```

---

## **Configuration**

The tool uses a YAML configuration file for flexible parameterization. Below is an example configuration:

```yaml
paths:
  raw_data: ./data/raw_data.csv
  processed_data: ./data/processed_data.csv
  figure: ./figures

parameters:
  data_type: raw
  correlation_threshold: 0.8
  target_variable: target

model_pars:
  C_reg: 1.0
  epsilon: 0.1
  n_components: 3

train_pars:
  test_size: 0.2
```

---

## **Logging**
Logs are configured to display key information about the tool's execution. Logs include timestamps, log levels, and messages.

Sample log format:
```
2024-12-19 09:10:00 - INFO - Data loaded successfully.
2024-12-19 09:15:00 - INFO - SVR model trained successfully.
```

---

## **Project Structure**
```
project-directory/
|
├── main.py                     # Entry point for the CLI tool
├── pipeline/
│   ├── data_processor.py       # Data processing tasks
│   ├── trainer.py              # Model training tasks
│   ├── evaluator.py            # Model evaluation tasks
│
├── utils/
│   ├── load_config.py          # Load YAML configuration
│   ├── display.py              # Plotting utilities
│
├── data/                       # Raw and processed data
├── figures/                    # Saved plots
└── requirements.txt            # Dependencies
```

---

## **Future Enhancements**
- Add support for additional models (e.g., Random Forest, Gradient Boosting).
- Implement hyperparameter tuning.
- Add functionality for cross-validation.

---

## **License**
This project is licensed under the MIT License.
