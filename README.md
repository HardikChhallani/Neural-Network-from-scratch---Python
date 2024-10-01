# Simple Neural Network for Predicting Insurance Purchase

This repository contains a simple implementation of a neural network from scratch, using Python and NumPy. The network predicts whether someone bought insurance based on their age and affordability.

## Dataset

The dataset used in this example consists of three columns:
1. **Age**: The age of the individual.
2. **Affordability**: Whether the person finds the insurance affordable (1 = Yes, 0 = No).
3. **Bought Insurance**: The target column, indicating whether the person bought the insurance (1 = Yes, 0 = No).

### Sample Dataset:

![Dataset Sample](./image.png)

## Neural Network

This is a simple neural network with no hidden layers (i.e., a perceptron) that uses the **sigmoid activation function**. It predicts the probability of buying insurance and uses **gradient descent** to learn the optimal weights and bias.

### Features

- **Age** and **Affordability** are used as input features.
- **Bought Insurance** is the binary target.
- The model uses gradient descent for optimization and logistic regression for classification.
  
### Model Overview

- **Weights**: Initialized to `1`.
- **Bias**: Initialized to `0`.
- **Activation Function**: Sigmoid function to map any input between 0 and 1.
- **Loss Function**: Log-loss (binary cross-entropy).

### Code Overview

#### 1. Model Class: `MyNN`

- `fit(X, y, epochs, learning_rate)`: Trains the neural network.
- `predict(X_test)`: Predicts the class label for the test set.
- `get_gradient_descent()`: Performs gradient descent to update weights and bias.
- `sigmoid(x)`: Sigmoid function implementation.
- `log_loss(y_true, y_pred)`: Log-loss (binary cross-entropy) calculation.

#### 2. Training and Prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the dataset (from CSV file)
df = pd.read_csv('insurance_data.csv')

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],
                                                    df.bought_insurance,
                                                    test_size=0.2,  
                                                    random_state=25)

# Scale the features (age scaled)
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['affordibility'] = X_test_scaled['affordibility'] / 100

# Create and train the neural network
neural_network = MyNN()
neural_network.fit(X_train_scaled, y_train, epochs=8000, learning_rate=0.1)

# Make predictions
predictions = neural_network.predict(X_test)
```

### Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

You can install the required libraries using the following:

```bash
pip install numpy pandas scikit-learn
```

### Running the Code

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies.
3. Run the Python script containing the neural network code.
4. Check the console output for the final weights, bias, and loss during training.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---