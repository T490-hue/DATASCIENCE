import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class Perceptron:
    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        self.activation_function=self._unit_step_function

    def fit(self,X,y):

        self.weights=np.zeros(X.shape[1])
        self.bias=0

        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predicted=self.activation_function(linear_output)

                update=self.learning_rate*(y[idx]-y_predicted)
                self.weights+=update*x_i
                self.bias+=update
    
    def predict(self,X):
        linear_output=np.dot(X,self.weights)+self.bias
        y_predicted=self.activation_function(linear_output)
        return y_predicted
    
    def _unit_step_function(self,z):
        return np.where(z>=0,1,0)


X,y=make_classification(n_samples=500,n_features=3,n_classes=2,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
perceptron=Perceptron(learning_rate=0.002,n_iters=1000)
perceptron.fit(X_train,y_train)
y_pred=perceptron.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

