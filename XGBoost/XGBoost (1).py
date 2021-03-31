# XGBoost 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
from IPython import get_ipython


# Consider the following data.
class XGB:
    def __init__(self):
        age = [9, 23, 24, 27, 45, 56, 62, 71, 76, 82, 84, 95, 79, 29, 34, 35, 49]
        fees = [69,78, 83, 82, 113, 119, 137, 145, 147, 158, 204, 189, 99, 100, 118, 112, 117]
        self.df = pd.DataFrame(columns=['age', 'fees'])
        self.df.age = year
        self.df.fees =fees 
        print("DATA GIVEN :")
        print(self.df.head())

    def plot_data(self):
        plt.scatter(x=self.df.age, y=self.df.fees)
        plt.show()





    def perform_boosting(self):
        df1 = self.df

        for i in range(2):
            f = self.df.fees.mean()
            if(i > 0):
                self.df['f'+str(i)] = self.df['f'+str(i-1)] + \
                    self.df['h'+str(i)]
            else:
                self.df['f'+str(i)] = f
            self.df['y-f'+str(i)] = self.df.fees - self.df['f'+str(i)]
            splitIndex = np.random.randint(0, self.df.shape[0]-1)
            a = []
            h_upper = self.df['y-f'+str(i)][0:splitIndex].mean()
            h_bottom = self.df['y-f'+str(i)][splitIndex:].mean()
            for j in range(splitIndex):
                a.append(h_upper)
            for j in range(self.df.shape[0]-splitIndex):
                a.append(h_bottom)
            self.df['h'+str(i+1)] = a
        print("Dataset after 2 iterations: ")
        print(self.df.head())


#ffter 2 iterations

#continue to iterate for 100 times , You will see the Loss of MSE(Fi) constantly reducing by a huge margin

        for i in range(100):
            f = self.df.fees.mean()
            if(i > 0):
                self.df['f'+str(i)] = self.df['f'+str(i-1)] + \
                    self.df['h'+str(i)]
            else:
                self.df['f'+str(i)] = f
            self.df['y-f'+str(i)] = self.df.fees - self.df['f'+str(i)]
            splitIndex = np.random.randint(0, self.df.shape[0]-1)
            a = []
            h_upper = self.df['y-f'+str(i)][0:splitIndex].mean()
            h_bottom = self.df['y-f'+str(i)][splitIndex:].mean()
            for j in range(splitIndex):
                a.append(h_upper)
            for j in range(self.df.shape[0]-splitIndex):
                a.append(h_bottom)
            self.df['h'+str(i+1)] = a
        print("Dataset afte 100 iterations :")
        print(self.df.head())


#graph for Iteration number 1 , 10 and 99
# the loss keeps on decreasing and the model adapting to the dataset as the iteration tend to increase


    def plot_loss(self):
        plt.figure(figsize=(15, 10))
        plt.scatter(self.df.age, self.df.fees)
        plt.plot(self.df.age, self.df.f1, label='f1')
        plt.plot(self.df.age, self.df.f10, label='f10')
        plt.plot(self.df.age, self.df.f99, label='f99')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    my_xgb = XGB()
    my_xgb.plot_data()
    my_xgb.perform_boosting()
    my_xgb.plot_loss()
