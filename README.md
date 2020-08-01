# decision__tree_regresor_visualisation
#more is the maximum depth maximum will be the chance of better classifying
#importing libraries 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
#putting the values to visulalise the working of decision tree regressor at values 7 and 2
value1 = 7
value2 = 2
#creatinng a random_datasets for the classifier
def create_datasets():
    rng = np.random.RandomState(1)
    x = np.arange(-10,10,0.4).reshape(-1,1)
    y = np.sin(x).ravel()
    return x,y
#function to compare the maximum depth of the regressor
def max_depth_comparison(value1,value2):
    x,y = create_datasets()
    regr1 = DecisionTreeRegressor(max_depth=value1)
    regr2 = DecisionTreeRegressor(max_depth=value2)
    regr1 = regr1.fit(x,y)
    regr2 = regr2.fit(x,y)
    return regr1,regr2
#predicting  values of the two regressor in case of two maximum depth
def predict(value1,value2):
    regr1,regr2 = max_depth_comparison(value1,value2)
    X_test = np.arange(0.0,5.0,0.01).reshape(-1,1)
    y1 = regr1.predict(X_test)
    y2 = regr2.predict(X_test)
    return X_test,y1,y2
#function to plot and visualise the data and our regressor line
def plot(value1,value2):
    x,y = create_datasets()
    X_test,y1,y2 = predict(value1,value2)
    plt.figure()
    plt.scatter(x,y,label='scatter_plot_of_initial_data',color='orange')
    plt.plot(X_test,y1,label='regressor1_plot',color='red')
    plt.plot(X_test,y2,label='regressor2_plot',color='blue')
    plt.show()
#plotting the individual datas for more clarification
def plot_individual(value1,value2):
    x,y = create_datasets()
    X_test,y1,y2 = predict(value1,value2)
    plt.subplot(311)
    plt.plot(x,y)
    plt.subplot(312)
    plt.plot(X_test,y1)
    plt.subplot(313)
    plt.plot(X_test,y2)
    plt.show()
        
    
