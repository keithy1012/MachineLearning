import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

class LogisticRegression:

    def __init__(self, lr, iters):
        self.lr = lr
        self.iters = iters


    def fit(self, x, y):
        # X is a matrix of mxn, m training samples, n features
        self.x = x
        self.m, self.n = x.shape
        # Y is a matrix of 1xm, m training classifications
        self.y = y.flatten()
        # w is the weights matrix
        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.iters):
            self.update_weights()
        
        return self
    
    def update_weights(self):
        expected = self.sigmoid(self.x.dot(self.w) + self.b)

        error = expected - self.y
        dW = np.dot(self.x.T, error) / self.m
        db = np.sum(error) / self.m
        self.w -= self.lr * dW
        self.b -= self.lr * db

        return self


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X, classification_param=0.5):
        expected = self.sigmoid(X.dot(self.w) + self.b)
        return (expected >= classification_param).astype(int)
        
def main() : 
      
    # Importing dataset     
    df = pd.read_csv( "diabetes.csv" ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values 
      
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 ) 
      
    # Model training     
    model = LogisticRegression(0.001, 10000) 
      
    model.fit( X_train, Y_train )     
    #model1 = LogisticRegression()     
    #model1.fit( X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict( X_test , 0.5)     
    #Y_pred1 = model1.predict( X_test ) 
      
    # measure performance     
    correctly_classified = 0    
    correctly_classified1 = 0
      
    # counter     
    count = 0    
    for count in range( np.size( Y_pred ) ) :   
        
        if Y_test[count] == Y_pred[count] :             
            correctly_classified = correctly_classified + 1
          
        #if Y_test[count] == Y_pred1[count] :             
        #    correctly_classified1 = correctly_classified1 + 1
              
        count = count + 1
          
    print( "Accuracy on test set by our model       :  ", (  
      correctly_classified / count ) * 100 ) 
    #print( "Accuracy on test set by sklearn model   :  ", (  
    #  correctly_classified1 / count ) * 100 ) 
  
  
if __name__ == "__main__" :      
    main()