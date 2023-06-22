import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
# Load the CSV file
cat = pd.read_csv("C:\CODING\DataScience\cat.csv")
dog = pd.read_csv("C:\CODING\DataScience\dog.csv")
dog = dog.T
cat = cat.T#.T is used for to get the transpose of the matrix
#Transpose means interchanging the rows and columns
#Rows = records and columns should be features(here features are the pixels).
#In the dog and cat dataset the rows were the pixels and columns were record so we had to transpose the dataset
print(cat.shape)
print(dog.shape)
#the .shape is used to find the dimensions of the matrix so it return the number of rows and columns in a tuple format


#iloc function is used to retrieve a specifc data from a dataframe 
# its syntax is = df.iloc[row_index,column_index]

'''
In Python, the .reshape() function is used to change the shape or dimensions of an array or matrix without altering its data.
 It allows you to reorganize the elements of the array into a new shape while preserving the original values.

The general syntax of .reshape() is arr.reshape(new_shape), where arr represents the array or matrix you want to reshape,
 and new_shape specifies the desired dimensions of the reshaped array.
'''
##plotting the first dog image
a = dog.iloc[79].values
#.values puts the pixel values in a numpy array 'a'
plt.imshow(a.reshape(64,64).T,cmap="gray")
#now the a.reshape will put the pixel values in a 64 rows and 64 columns to get a visual represenetation
#we can rotate the image using by transposing the matrix hence we use .T
#plt.imshow is used to display a image with specific colormap
    


#now we will concat the dog and cat dataset into one so that we can do training and testing together

total = pd.concat([dog,cat])
pca = PCA(0.95)#to get 95 percent variance pca is an object of class PCA
pca_total = pca.fit_transform(total)#calling function fit_transform  to find get the Principal Components
print(pca_total.shape)  


'''
in the above code pca = PCA(0.95) it will capture 95% variance in the data
but if we wrote pca = PCA(n_components=100) is will produce 100 principal components which will capture the variance
'''
#convert the pca_total into a dataframe
pca_total = pd.DataFrame(pca_total)
print(pca_total.head)


#finding the pca components

components = pca.components_
components=pd.DataFrame(components)
print(components.shape)
#here each row will be a principal component and each column will be the feature here the pixels s
#so the .shape will print 78 and 4096


#now lets plot the image using the pca

#plotting the first component
plt.imshow(components.iloc[0].values.reshape(64,64).T,cmap="gray")
#Plotting the first component. Here we can see that it captures the most common feature
# present in both dogs and cats hence the image looks like a combination of both dog and cat face.

#now lets try to reconstruct the dog face
'''
We had the pca _total which is a trasnformed dataset after applying the PCA
if we want to reconstruct the data to its original form we will perform 
inverse_transform function(pca_total) 
this will recontruct the pca_total to possible as there will be loss of features due to 
dimentionality reduction

inverse_pca=pca.inverse_transoform(pca_total)  which has only 95% variance
and then we put it into a dataframe.
'''
inverse_pca = pca.inverse_transform(pca_total)
inverse_pca = pd.DataFrame(inverse_pca)
inverse_pca.head()

#now we can compare the original image to the image which has 95% variance
plt.figure(figsize=(15,15))#to apply the image size #figsize=(width,height)
plt.subplot(2,2,1)
plt.title('Projected Face')
plt.imshow(inverse_pca.iloc[3].values.reshape(64,64).T,cmap='gray')
plt.title('Projected Face')
plt.imshow(inverse_pca.iloc[3].values.reshape(64,64).T,cmap='gray')
plt.xticks([])#to remove the x axis lines
plt.yticks([])#to remove the y axis lines
plt.subplot(2,2,2)
plt.title('Original Face')
plt.imshow(dog.iloc[3].values.reshape(64,64).T,cmap='gray')
plt.xticks([])
plt.yticks([])



#applying logistic regression  to classify into cats and dogs

log = LogisticRegression()
print("The total dataset is :")

#total = total.reset_index()#reset the label to 0 if any 
#also there will be an extra feature due to it as it puts the original labels into a
#new column called labels
total['index'] = 0# creatatd a new column named table and assigned 0 to all values(dummy values)
print(total.shape)


#As there is no label in the dataset hence we set the labels for the 
#classes as we know when we combined the cat and dog dataframe so first 80 observations 
#are dogs and next 80 observations are cat hence we label the dog class as 1 and cats class as 0.

total['index'][:80]=1
total['index'][80:]=0

#dividing the dataset into x and y
x= total.drop('index',1)# remove the column named index form the dataset 'total' 1- for column and 0 - for row
y= total['index'] #this is the target variable



'''
Note : When using logistic regression with scikit-learn, the input features (x) 
should be a 2D array or a pandas DataFrame, and the target variable (y) should be a 1D array or a pandas Series
so we the code x_train is a 2d array but we had to convert the y_train and y_test to a 1 D array
so we used the .ravel() function to convert to 1D array.
.values was used to get the data of the dataframe into a numpy array 
so we use the 2 function together
y_test=y_test.values.ravel()
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()#converts multidimensional array to 1d array    
y_test=y_test.values.ravel()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#fitting the dataset into logistics regression
log.fit(x_train, y_train)

y_pred=log.predict(x_test)
# use the confusion matrix to get the accuray of the model  and evaluate the perfomance of the classification model
confusion_matrix(y_test, y_pred)
print('Accuracy score for the testing values is ',accuracy_score(y_test,y_pred))


