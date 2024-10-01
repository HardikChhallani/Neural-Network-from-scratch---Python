import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self,X,y,epochs,learning_rate):
        self.w1, self.w2, self.bias = self.get_gradient_descent(X['age'],X['affordibility'],y,epochs,learning_rate)
        print(f"\nFinal weights and bias: \n w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")

    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordibility'] + self.bias
        return self.sigmoid(weighted_sum)

    def get_gradient_descent(self,age,affordibility,y,epochs,learning_rate):
        w1 = self.w1
        w2 = self.w2
        bias = self.bias
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1*age + w2*affordibility + bias
            y_predicted = self.sigmoid(weighted_sum)
            loss = self.log_loss(y,y_predicted)

            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y)) 
            w2d = (1/n)*np.dot(np.transpose(affordibility),(y_predicted-y)) 
            bias_d = np.mean(y_predicted-y)

            w1 = w1 - learning_rate * w1d
            w2 = w2 - learning_rate * w2d
            bias = bias - learning_rate * bias_d
            
            if i%5==0:
                print (f'Epoch:{i}, w1:{self.w1}, w2:{self.w2}, bias:{self.bias}, loss:{loss}')

        return w1, w2, bias


    def log_loss(self,y_true, y_predicted):
        epsilon = 1e-15
        y_predicted_new = [max(i,epsilon) for i in y_predicted]
        y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
        y_predicted_new = np.array(y_predicted_new)
        return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

    def sigmoid(self,x):
        x = np.clip(x, -500, 500)
        return 1/1+np.exp(-x)

df = pd.read_csv('insurance_data.csv')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],
                                                    df.bought_insurance,
                                                    test_size=0.2,  
                                                    random_state=25)

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['affordibility'] = X_test_scaled['affordibility'] / 100

neural_network = MyNN()
neural_network.fit(X_train_scaled, y_train, epochs=80, learning_rate=0.1)

neural_network.predict(X_test)