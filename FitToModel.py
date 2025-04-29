import numpy as np
import matplotlib.pyplot as plt
import DataProcessing

class Model:
    def __init__(self, scalar=1, degree=4, learning_rate=0.01, tolerance = 1e-6):
        self.scalar = scalar
        self.degree = degree
        x_cords = np.linspace(0, 40, 40)
        x_cords = x_cords.clip(min=0.001)
        self.x_cords = np.log(x_cords)
        # self.x_cords = (1 / x_cords[x_cords.size - 1]) * x_cords
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def pred(self):
        return self.scalar + self.degree * self.x_cords

    def meanSquaredError(self, y_true):
        y_pred = self.pred()
        return np.mean((y_true - y_pred) ** 2)
    
    def scalarGrad(self, y_true):
        y_pred = self.pred()
        return np.mean(2 * (y_pred - y_true))

    def degreeGrad(self, y_true):
        y_pred = self.pred()
        return np.mean(2 * (y_pred - y_true) * self.x_cords)
    
    def reset(self):
        self.scalar = 1
        self.degree = 1/2
        x_cords = np.linspace(0, 40, 40)
        x_cords = x_cords.clip(min=0.001)
        self.x_cords = np.log(x_cords)
        # self.x_cords = (1 / x_cords[x_cords.size - 1]) * x_cords
        self.learning_rate = 0.01
        self.tolerance = 1e-6


    def fit(self, y_true, plot=False):
        y_true = np.log(y_true.clip(min=0.01))
        degree_old = self.degree - 2 * self.tolerance
        scalar_old = self.scalar - 2 * self.tolerance

        while (abs(self.degree - degree_old) > self.tolerance or abs(self.scalar - scalar_old) > self.tolerance):
            degree_old = self.degree
            scalar_old = self.scalar
            
            self.degree -= self.learning_rate * self.degreeGrad(y_true)
            self.scalar -= self.learning_rate * self.scalarGrad(y_true)


        if plot is True:
            # plt.plot(self.x_cords, y_true, label="True Values")
            # plt.plot(self.x_cords, self.pred(), label="Predictions")

            plt.plot(np.exp(self.x_cords), np.exp(y_true), label="True Values")
            plt.plot(np.exp(self.x_cords), np.exp(self.pred()), label="Predictions")

            plt.legend()
            plt.show()
        
        return self.meanSquaredError(y_true) 
        

model = Model()

num = 100
data = DataProcessing.X[0:num, 10:50]

model.fit(data[39], True)

error = np.zeros(data.shape[0])
for i in range(0, error.shape[0]):
    model.reset()
    error[i] = model.fit(data[i, :])

error[error >= 0.5] = 1
error[error < 0.5] = 0
error = error.round()

true_error = DataProcessing.y[0:num]

print(error)
print(true_error)

for i in range(0, error.shape[0]):
    if error[i] != true_error[i]:
        print(i)

