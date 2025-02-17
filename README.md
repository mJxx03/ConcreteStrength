# Predictive Modeling of Concrete Compressive Strength

This project focuses on developing and evaluating a neural network model using the Keras library to predict the compressive strength of concrete based on its material composition. The model underwent multiple phases of experimentation, with variations in data preprocessing, training epochs, and network architecture to analyze performance improvements and optimize predictive accuracy.

## Usage

To execute the model, follow these steps:

### 1. Open in Jupyter Notebook
Open the provided Jupyter notebook to access the code and analysis workflow.

### 2. Install Dependencies
Run the initialization cells to install essential Python libraries, including Keras, TensorFlow, NumPy, Pandas, and Scikit-Learn.

### 3. Prepare Dataset
- Import the dataset containing concrete composition and strength measurements.
- Split the data into training and testing sets using an 80-20 ratio.

### 4. Train Baseline Model
- Build a neural network with a single hidden layer comprising 10 neurons and ReLU activation.
- Train the model for a specified number of epochs using the Adam optimizer and mean squared error (MSE) as the loss function.
- Evaluate performance by calculating MSE across multiple runs to analyze model stability.

### 5. Data Normalization
- Apply normalization techniques by scaling features to a standard distribution.
- Retrain the model and compare results with the baseline configuration.

### 6. Epoch Adjustment
- Increase the number of training epochs to assess the impact of extended training duration on performance metrics.

### 7. Architectural Complexity
- Modify the model to include three hidden layers, each with 10 neurons.
- Retrain and evaluate to compare performance against the simpler architecture.

## Performance Evaluation

Performance was assessed using the following metrics:
- **Mean Squared Error (MSE)**: Primary metric for evaluating predictive accuracy.
- **Standard Deviation of MSE**: Analyzed to assess model stability across multiple training cycles.

The results were documented to compare the impact of different preprocessing techniques, epoch counts, and architectural adjustments.

## Code Overview
- **ConcreteStrengthPrediction.ipynb**: Main Python notebook containing the code and analysis.
- **requirements.txt**: List of required Python packages.
- **data.csv**: Dataset containing concrete composition details and strength measurements.

## Dependencies

**Python Libraries:**
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

## Project Insights

The project successfully demonstrated the influence of data normalization, increased training epochs, and network depth on predictive performance. The findings highlighted the importance of feature scaling and model complexity in improving neural network accuracy for material strength prediction.
