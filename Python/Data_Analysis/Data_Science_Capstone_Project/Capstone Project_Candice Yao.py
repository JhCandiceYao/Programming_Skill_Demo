# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:41:22 2022

@author: Candice Yao
"""
#%%import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression 
from scipy.special import expit # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#%% Load data
#(1) as a numpy array
arr_movie = np.genfromtxt('movieReplicationSet1.csv',delimiter=',',skip_header=True)
#(2) as a dataframe
df_movie = pd.read_csv('movieReplicationSet1.csv', encoding='latin-1',header=None)
#%% question1 What is the relationship between sensation seeking and movie experience?
#do pca in each categories and compute their correlation
#the first 20 vars(columns) describe sensation seeking questions and the last 10 vars(columns) describe movie experience. 
df_sens_mov = df_movie.iloc[1:, np.r_[400:420,464:474]]#total dataframe
#since there are a large numebr of n, row-wise(paritipant wise) dropping of na will not largely alter the result
df_sens_mov = df_sens_mov.dropna()
#store seperate dataframe without nans
df_sens = df_sens_mov.iloc[:,0:20].astype(float)
df_mov_exp =  df_sens_mov.iloc[:,20:30].astype(float)
#investigate correlaiton in each of the category of questions
corr_sens = df_sens.corr()
plt.imshow(corr_sens) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.colorbar()
plt.title(label="Correlation between sensation seeking questions") 
plt.show()
#there is some degree of correlation within the questions of sensation seekings
#invoke PCA to reduce dimension
#1. PCA for sensation 
#step 1 normalize the data
scaler = preprocessing.StandardScaler()
scaled_sens = scaler.fit_transform(df_sens)
pca = PCA().fit(scaled_sens)
#3a eigenvalues
eig_vals = pca.explained_variance_
#3b eigenvectors(specific weighs shared by each original vars in the PCs)
loadings = pca.components_
#3c
rotated_sens = pca.fit_transform(df_sens)
#display how much each PC variance is explained by each PC
var_explained = eig_vals/sum(eig_vals)*100
# display this for each factor:
for ii in range(len(var_explained)):
    print(var_explained[ii].round(3))
#scree plot
num_questions = len(df_sens.columns)
x = np.linspace(1,num_questions,num_questions)
plt.bar(x, eig_vals, color='gray')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title(label="Scree plot of sensation seeking questions") 
plt.show()
#check loadings 
x1 = np.linspace(1,num_questions,num_questions)
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x1,loadings[whichPrincipalComponent,:]*(-1)) #eigenvestors multiplied by -1 because the direction is arbitrary
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title(label="loadings of the first PC of sensation questions") 
plt.show() # Show bar plot

#check movie experience
corr_mov_exp = df_mov_exp.corr()
plt.imshow(corr_mov_exp) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.colorbar()
plt.title(label="Correlation between movie experience questions") 
plt.show()
#there is some degree of correlation within the questions of movie experience 
#invoke PCA to reduce dimension
#PCA for movie experience questions
#step 1 normalize the data
scaler = preprocessing.StandardScaler()
scaled_exp = scaler.fit_transform(df_mov_exp)
pca = PCA().fit(scaled_exp)
#3a eigenvalues
eig_vals = pca.explained_variance_
#3b eigenvectors(specific weighs shared by each original vars in the PCs)
loadings = pca.components_
#3c
rotated_exp = pca.fit_transform(df_mov_exp)
#display how much each PC variance is explained by each PC
var_explained = eig_vals/sum(eig_vals)*100
# display this for each factor:
for ii in range(len(var_explained)):
    print(var_explained[ii].round(3))
#scree plot
num_questions = len(df_mov_exp.columns)
x = np.linspace(1,num_questions,num_questions)
plt.bar(x, eig_vals, color='gray')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title(label="Scree plot of movie experience questions") 
plt.show()
#check loadings 
x1 = np.linspace(1,num_questions,num_questions)
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x1,loadings[whichPrincipalComponent,:]*(-1)) #eigenvestors multiplied by -1 because the direction is arbitrary
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title(label="loadings of the first PC of movie experience questions") 
plt.show() # Show bar plot
#take the first PC of sensation and movie experience and compute their correlation 
sens = rotated_sens[:,0]
mov_exp = rotated_exp[:,0]
corr_sens_exp = np.corrcoef(sens, mov_exp)
print(corr_sens_exp)
# visualize the new data using the two pc columns
plt.plot(sens,mov_exp,'o',markersize=2)
plt.xlabel('sensasion seeking')
plt.ylabel('movie experience')
plt.title('sensation seeking vs. movie experience')
#%%question2 Is there evidence of personality types based on the data of these research participants?
# If so, characterize these types both quantitatively and narratively
df_personality =df_movie.iloc[1:, 420:464].astype(float)
print(df_personality.info)
# to ascertain whether a PCA is indivated
corr_pers = df_personality.corr()
plt.imshow(corr_pers,cmap='coolwarm')
plt.colorbar()
plt.title(label='Correlation between personality questions')
plt.show()
# analysis: they are not uncorrelated. visually, there are some light red dots that indicate colinearities between certain questions
#1a. normalize the data
df_personality = df_personality.dropna()
scaler = preprocessing.StandardScaler()
scaled_personality = scaler.fit_transform(df_personality)
#PCA
pca_pers = PCA().fit(scaled_personality)
eig_vals_pers = pca_pers.explained_variance_
loadings_pers = pca_pers.components_
rotated_pers = pca.fit_transform(scaled_personality)
#display how much each PC variance is explained by each PC
var_explained_pers = eig_vals_pers/sum(eig_vals_pers)*100
# display this for each factor:
for jj in range(len(var_explained_pers)):
    print(var_explained_pers[jj].round(3))

#scree plot
num_questions_pers = len(df_personality.columns)
x = np.linspace(1,num_questions_pers,num_questions_pers)
plt.bar(x, eig_vals_pers, color='gray')
plt.plot([0,num_questions_pers],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title(label='Scree plot of personality questions')
plt.show()

threshold = 1 #Kaiser
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eig_vals_pers > threshold))
#Number of factors selected by Kaiser criterion: 8
#according to kaiser criterion, we should kept first eight princple components

#investigate the loadings 
num_factor = 8 #accordng to Kaiser threshold
for ii in range(num_factor):
    plt.bar(x,loadings_pers[ii,:]*-1, color = 'gray') # note: eigVecs multiplied by -1 because the direction is arbitrary
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.title(label="Loadings for PC" + str(int(ii+1)))
    plt.show() # Show bar plot
#investigate the loadings of the first 2 
#========

#plot the data using the rotated coordinate system, with the first two pcs as x and y
plt.plot(rotated_pers[:,0],rotated_pers[:,1],'o',markersize=2)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title(label='PCA-rotated personality data')
plt.show() 

#the actual clustering using K-means
#1 store transformed data 
#according to kaiser we keep 8 pcs
pers_transformed = rotated_pers[:,:2]

#using Silhouette to check the optimal number of clusters:
#Init:
num_clusters = 9 # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([num_clusters,1])*np.NaN # init container to store sums
# Compute kMeans for each k:
for ii in range(2, num_clusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(pers_transformed) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(pers_transformed,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) 
    plt.tight_layout() 
    
# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,num_clusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()   
# the Silhouette score is the highest at k = 2 

# 2j) Now that we determined the optimal k, we can now ask kMeans to cluster the data for us, 
num_clusters = 2
kMeans = KMeans(n_clusters = num_clusters).fit(pers_transformed) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 
# Plot the color-coded data:
for ii in range(num_clusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(pers_transformed[plotIndex,0],pers_transformed[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('social hostility')
    plt.ylabel('intellectual laziness')

#print out the questions title for in-depth grouping and naming the new variables(PCs)
questions = df_movie.iloc[0,:]

print(questions)
questions_personality = questions.columnsc[421:465]
print(questions_personality)

#%%question3 Are movies that are more popular rated higher than movies that are less popular?
#*You can operationalize the popularity of a movie by how many ratings it has received. 

#slice out all the columns that contain movie ratings
arr_ratings = arr_movie[:,0:400].T #transpose it so each movie is a row and the columns are the variables that charaterize each movie
rat_pop = np.zeros([400,2])#initialize a new 2d table where each row is one movie, the 2 columns represent respectively the ratings and the popularity
rat_pop[:] = np.NaN
num_ratings = np.array([])#create an array that stores the number of ratings of each movie
mean_ratings = np.array([])#create an array that stores the average ratings of each movie
for aa in range(len(arr_ratings)):
    each_movie = arr_ratings[aa,:] #the slice of ratings of each movie
    each_movie = each_movie[np.isfinite(each_movie)]
    num_ratings = np.append(num_ratings, len(each_movie))
    each_rate_mean = np.nansum(arr_ratings[aa,:])/num_ratings[aa]
    mean_ratings = np.append(mean_ratings, each_rate_mean)
rat_pop[:,0] = num_ratings.T
rat_pop[:,1] = mean_ratings.T

x = rat_pop[:,0].reshape(len(rat_pop[:,0]),1)
y = rat_pop[:,1]
corr_rat_pop = np.corrcoef(x,y,rowvar= False)

#using SLR to fit a line to X and Y
model_rat_pop = LinearRegression().fit(x,y)
r_sq1 = model_rat_pop.score(x, y)
slope1 = model_rat_pop.coef_
intercept1 = model_rat_pop.intercept_
print(slope1)
print(r_sq1)
yHat1 = slope1*rat_pop[:,0] + intercept1
#plot the finished rat_pop table to visualize the relationship between popularity and ratings
plt.plot(rat_pop[:,0],rat_pop[:,1],'o',markersize=2)
plt.xlabel('popularity')
plt.ylabel('ratings')
plt.title('popularity vs ratings')
plt.plot(rat_pop[:,0], yHat1, color='purple', linewidth=3)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(r_sq1))

plt.imshow(corr_rat_pop) 
plt.colorbar()
plt.title(label="Correlation between ratings and popularity) 
plt.show()

plt.plot(rat_pop[:,0], yHat1, color='purple', linewidth=3)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(r_sq1))
#%%question4 Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
#locate the index  of Shrek
header = df_movie.iloc[0,0:400] #save a dataframe that shows all the name of the columns
header_arr = np.array(header)#save list of column names as an array
Shrek_index = np.where(header_arr == 'Shrek (2001)')
print(Shrek_index) 
#(array([87], dtype=int64)
#proprocessing data: dropping participants that did not rate this movie:
M1 = arr_movie[:,87] #ratings
M2 = arr_movie[:,474] #gender
#dropping nans
temp = np.array([np.isnan(M1), np.isnan(M2)], dtype = bool)
temp2 = temp * 1
temp3 = sum(temp2)
missing_data = np.where(temp3 > 0)
M1 = np.delete(M1, missing_data)
M2 = np.delete(M2, missing_data)
#combine the column of shrek ratings with the column of gender into an array
shrek_gender = np.transpose(np.array([M1,M2]))
#split female and male ratings
female_index = np.where(shrek_gender[:,1] == 1)
female_ratings = shrek_gender[female_index]
female_ratings = female_ratings[:,0]
male_index = np.where(shrek_gender[:,1] == 2)
male_ratings = shrek_gender[male_index]
male_ratings = male_ratings[:,0]

#visualize 
df_female = pd.DataFrame(female_ratings)
df_male = pd.DataFrame(male_ratings)
plt.boxplot(df_female,showmeans=True)
plt.title('female ratings of Shrek')
plt.boxplot(df_male,showmeans=True,)
plt.title('male rating of Shrek')

#Mann Whitney U test since not sure if data have linear properties 
u1,p1 = stats.mannwhitneyu(female_ratings, male_ratings)
print(u1,p1)
#%% question5 Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
#locate the index  of The Lion King
Lion_index = np.where(header_arr == 'The Lion King (1994)')
print(Lion_index) 
#(array([220], dtype=int64)
#proprocessing data: dropping participants that did not rate this movie:
M1 = arr_movie[:,220] #ratings of The Lion King
M2 = arr_movie[:,475] #family situation: only child or not 
#dropping nans
temp = np.array([np.isnan(M1), np.isnan(M2)], dtype = bool)
temp2 = temp * 1
temp3 = sum(temp2)
missing_data = np.where(temp3 > 0)
M1 = np.delete(M1, missing_data)
M2 = np.delete(M2, missing_data)
#remove all the rows where there is no response about family situation
no_response_index = np.where(M2 == -1)
M1 = np.delete(M1, no_response_index)
M2 = np.delete(M2, no_response_index)
#combine the column of shrek ratings with the column of gender into an array
Lion_child = np.transpose(np.array([M1,M2]))
#split people who are only child and those who have sibling
only_index = np.where(Lion_child[:,1] ==1)
sib_index = np.where(Lion_child[:,1] ==0)
only_ratings = Lion_child[only_index]
only_ratings = only_ratings[:,0]
sib_ratings = Lion_child[sib_index]
sib_ratings = sib_ratings[:,0]
#visualize 
df_only = pd.DataFrame(only_ratings)
df_sib = pd.DataFrame(sib_ratings)
plt.boxplot(df_only,showmeans=True)
plt.title('only child ratings of The Lion King')
plt.boxplot(df_sib,showmeans=True)
plt.title('people with siblings ratings of The Lion King')

#Mann Whitney U test since not sure if data have linear properties 
u2,p2 = stats.mannwhitneyu(only_ratings, sib_ratings)
print(u2,p2)
#52929.0 ;
#0.04319872995682849
#%% question6 Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than
# those who prefer to watch them alone?
#locate the index  of ‘The Wolf of Wall Street (2013)'
Wolf_index = np.where(header_arr == 'The Wolf of Wall Street (2013)')
print(Wolf_index) 
#(array([357], dtype=int64)
#proprocessing data: dropping participants that did not rate this movie:
M1 = arr_movie[:,357] #ratings of Wolf
M2 = arr_movie[:,476] #social viewing preference 
#dropping nans
temp = np.array([np.isnan(M1), np.isnan(M2)], dtype = bool)
temp2 = temp * 1
temp3 = sum(temp2)
missing_data = np.where(temp3 > 0)
M1 = np.delete(M1, missing_data)
M2 = np.delete(M2, missing_data)
#remove those who didn't report their social viewing preference 
no_response_index = np.where(M2 == -1)
M1 = np.delete(M1, no_response_index)
M2 = np.delete(M2, no_response_index)
#combine the column of wolf ratings with the column of social viewing preference
wolf_social = np.transpose(np.array([M1,M2]))
#split those who love to view alone and those who don't 
alone_index = np.where(wolf_social[:,1] == 1)
alone_ratings = wolf_social[alone_index]
alone_ratings = alone_ratings[:,0]
notalone_index = np.where(wolf_social[:,1] == 0)
notalone_ratings = wolf_social[notalone_index]
notalone_ratings = notalone_ratings[:,0]

#visualize 
df_alone = pd.DataFrame(alone_ratings)
plt.boxplot(df_alone,showmeans=True)
plt.title('people who enjoy movies alone ratings of The Wolf of Wall Street')
plt.show()
df_notalone = pd.DataFrame(notalone_ratings)
plt.boxplot(df_notalone,showmeans=True)
plt.title('people who do not enjoy movies alone ratings of The Wolf of Wall Street')
plt.show()

#Mann Whitney U test since not sure if data have linear properties 
u3,p3 = stats.mannwhitneyu(alone_ratings, notalone_ratings)
print(u3,p3)
#56806.5 
#0.1127642933222891
#%% question7 There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’,
#‘Indiana Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this
# dataset. How many of these are of inconsistent quality, as experienced by viewers?
movie_with_title = pd.read_csv('movieReplicationSet1.csv', encoding='latin-1') #store a dataframe where pd
print(movie_with_title.columns)
movielist = ['Star Wars', 'Harry Potter', 'The Matrix','Indiana Jones', 'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']
movie_dict = {"SW": np.array([]),
              "HP" : np.array([]),
              "TM" : np.array([]),
              "IJ" : np.array([]),
              "JP" : np.array([]),
              "PC" : np.array([]),
              "TS" : np.array([]),
              "BM" : np.array([])}


for i, j in zip(range(8),movie_dict.keys()):
    x = movielist[i]
    index = movie_with_title.columns.str.contains(x)
    y = movie_with_title[movie_with_title.columns[index]]
    y = y.dropna()
    movie_dict[j] = y
SW = movie_dict["SW"]
HP = movie_dict["HP"]  
TM = movie_dict["TM"]
IJ = movie_dict["IJ"]
JP = movie_dict["JP"]
PC = movie_dict["PC"]
TS = movie_dict["TS"]
BM = movie_dict["BM"]

h,pK = stats.kruskal(SW.iloc[:,0].to_numpy().flatten(), SW.iloc[:,1].to_numpy().flatten(),SW.iloc[:,2].to_numpy().flatten())
print(h,pK)
h1,pK1 = stats.kruskal(HP.iloc[:,0].to_numpy().flatten(), HP.iloc[:,1].to_numpy().flatten(), HP.iloc[:,2].to_numpy().flatten(),HP.iloc[:,3].to_numpy().flatten())
print(h1,pK1)
h2,pK2 = stats.kruskal(TM.iloc[:,0].to_numpy().flatten(), TM.iloc[:,1].to_numpy().flatten(), TM.iloc[:,2].to_numpy().flatten())
print(h1,pK2)
h3,pK3 = stats.kruskal(IJ.iloc[:,0].to_numpy().flatten(), IJ.iloc[:,1].to_numpy().flatten(), IJ.iloc[:,2].to_numpy().flatten(),IJ.iloc[:,3].to_numpy().flatten())
print(h3,pK3)
h4,pK4 = stats.kruskal(JP.iloc[:,0].to_numpy().flatten(), JP.iloc[:,1].to_numpy().flatten(), JP.iloc[:,2].to_numpy().flatten())
print(h4,pK4)
h5,pK5 = stats.kruskal(PC.iloc[:,0].to_numpy().flatten(), PC.iloc[:,1].to_numpy().flatten(), PC.iloc[:,2].to_numpy().flatten())
print(h5,pK5)
h6,pK6 = stats.kruskal(TS.iloc[:,0].to_numpy().flatten(), TS.iloc[:,1].to_numpy().flatten(), TS.iloc[:,2].to_numpy().flatten())
print(h6,pK6)
h7,pK7 = stats.kruskal(BM.iloc[:,0].to_numpy().flatten(), BM.iloc[:,1].to_numpy().flatten(), BM.iloc[:,2].to_numpy().flatten())
print(h7,pK7)
#plot some of the movies to visualize the changes in ratings
plt.boxplot(BM,showmeans=True)
plt.show()
plt.boxplot(SW,showmeans=True)
plt.title("Star Wars movies' ratings")
plt.show()
plt.boxplot(HP,showmeans=True)
plt.title("Harry Potter movies' ratings")
plt.show()
#%% question8 Build a prediction model of your choice (regression or supervised learning) to predict movie
#ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation
#methods to avoid overfitting and characterize the accuracy of your model.
#Random Forest that uses all 44 personality questions to predict the rating of each movie

X = df_movie.iloc[:, 420:464] #all the personality question for every participant 
personality_median = np.median([1,2,3,4,5]) #use the median rating to replace nans
X = X.fillna(personality_median)
y = df_movie.iloc[:,0:400]
median_rating = y.median(axis=0)
for u in range(len(y.columns)):
    y.iloc[:,u] = y.iloc[:,u].fillna(median_rating[u]) #handling nans by personal median rating of a participant
    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.8)
print(len(X_train), len(Y_train),len(X_test),len(Y_test))#877 participants go into trainning set and 220 go into test set

print(Y_train.dtypes)
#since Y_train is now full of continuous data, the tree only classify discrete data, we need to preprocess Y_train to make it discrete
Y_train = Y_train *100#(get rid of the digit)
Y_test = Y_test*100.
Y_test = Y_test.to_numpy()

num_trees = 100#Large enough to invoke CLT
model_accuracy_all = np.zeros((400))
RMSE_all = np.zeros((400))
for i in range(len(Y_train.columns)):
    clf = RandomForestClassifier(n_estimators=num_trees).fit(X_train,Y_train.iloc[:,i]) #bagging numTrees trees for every of the 400 movies
    Y_predictions = clf.predict(X_test) 
    model_accuracy_all[i] = accuracy_score(Y_test[:,i], Y_predictions)
    RMSE_all[i] = sqrt(mean_squared_error(Y_test[:,i], Y_predictions)) 
#plot the first decision tree
fig = plt.figure(figsize=(150, 100))
plot_tree(clf.estimators_[0],fontsize=15,filled=True, impurity=True, 
          rounded=True)

average_accuracy = sum(model_accuracy_all)/len(Y_train.columns)
print(average_accuracy)
average_RMSE = sum(RMSE_all)/len(Y_train.columns)
print(average_RMSE)

plt.plot(model_accuracy_all,'o',markersize=3)
plt.xlabel('model of specific movie')
plt.ylabel('accuracy score')
plt.title('', kwargs)
plt.show() 

plt.plot(RMSE_all,'o',markersize=3, color= 'red')
plt.xlabel('model of specific movie')
plt.ylabel('RMSE')
plt.show() 


#try multiple regression
b0_all = np.zeros((len(Y_train.columns),len(Y_train)))
b1_all = np.zeros((len(Y_train.columns),len(Y_train)))   
RMSE_regress = np.zeros(len(Y_train.columns))

for j in range(len(Y_train.columns)):
    Model1 = LinearRegression().fit(X_train,Y_train.iloc[:,j])
    rSqrFull = Model1.score(X_train,Y_train.iloc[:,j])
    Y_predictions = Model1.predict(Y_test[:,j])
    Y_predictions = Y_predictions.reshape(-1,1)
    RMSE = sqrt(mean_squared_error(Y_test[:,j], Y_predictions))
    RMSE_regress[j] = RMSE
    print("rSqrFull for movie" + str(j) + "is", rSqrFull)
    print("RMSE for movie" + str(j) + "is", RMSE)
    b0 = Model1.intercept_.flatten()
    b1 = Model1.coef_.flatten()
    print(b0,b1)

#%% question9 Build a prediction model of your choice (regression or supervised learning) to predict movie
#random forest
#Initialize data
X = df_movie.iloc[:,474:478]#three columns of gender identity, sibship status, and social viewing preference
#preprocess X to get rid of participants with no answer in any of the question
# y is already preprocessed in the previous question
Xy = X.join(y,how='left')
Xy = Xy.dropna()#get rid of participants who didn't have answer for any of the question
#store X and y seperately
X = Xy.iloc[:,:3]#three columns of gender identity, sibship status, and social viewing preference
y = Xy.iloc[:,3:]#400 columns of movie ratings for each participants

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.8)
print(len(X_train), len(Y_train),len(X_test),len(Y_test))#858, 215 
print(Y_train.dtypes)
#since Y_train is now full of continuous data, the tree only classify discrete data, we need to preprocess Y_train to make it discrete
Y_train = Y_train *100#(get rid of the digit)
Y_test = Y_test*100.
Y_test = Y_test.to_numpy()

num_trees = 100#Large enough to invoke CLT
model2_accuracy_all = np.zeros((400))
RMSE2_all = np.zeros((400))
#loop a tree for every movie using the train set and test it with test set
for i in range(len(Y_train.columns)):
    clf = RandomForestClassifier(n_estimators=num_trees).fit(X_train,Y_train.iloc[:,i]) #bagging numTrees trees for every of the 400 movies
    Y_predictions = clf.predict(X_test) 
    model2_accuracy_all[i] = accuracy_score(Y_test[:,i], Y_predictions)
    RMSE2_all[i] = sqrt(mean_squared_error(Y_test[:,i], Y_predictions)) 

#plot the first decision tree out of the 100 random trees
fig = plt.figure(figsize=(50,50))
plot_tree(clf.estimators_[0],fontsize=15,filled=True, impurity=True, 
          rounded=True)
#calcualte overall performance across all movies' models
average_accuracy2 = sum(model2_accuracy_all)/len(Y_train.columns)
print(average_accuracy2)
average_RMSE2 =  sum(RMSE2_all)/len(Y_train.columns)
print(average_RMS2)

#visualize performance of each model
plt.plot(model2_accuracy_all,'o',markersize=3)
plt.xlabel('model of specific movie')
plt.ylabel('accuracy score')
plt.show() 

plt.plot(RMSE2_all,'o',markersize=3, color= 'red')
plt.xlabel('model of specific movie')
plt.ylabel('RMSE')
plt.show() 

#Confusion Matrix 
#Plot sample confusion matrix for the first movie
matrix2 = confusion_matrix(Y_test[:,400], Y_predictions[])
#%%question10  Build a prediction model of your choice (regression or supervised learning) to predict movie
#ratings (for all 400 movies) from all available factors that are not movie ratings (columns 401-
#477). Make sure to use cross-validation methods to avoid overfitting and characterize the
#accuracy of your model.

X = df_movie.iloc[:,400:478]#all columns other than movie ratings
# preprocess y by imputing missing ratings with the median rating of that movie.
y = df_movie.iloc[:,0:400]
median_rating = y.median(axis=0)
for u in range(len(y.columns)):
    y.iloc[:,u] = y.iloc[:,u].fillna(median_rating[u]) 
Xy = X.join(y,how='left')
Xy = Xy.dropna()#get rid of participants with null value 
#store X and y seperately
X = Xy.iloc[:,:77]#all columns other than movie ratings
y = Xy.iloc[:,77:]#400 columns of movie ratings for each participants


X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.8)
print(len(X_train), len(Y_train),len(X_test),len(Y_test))#756, 190 

print(Y_train.dtypes)
#since Y_train is now full of continuous data, the tree only classify discrete data, we need to preprocess Y_train to make it discrete
Y_train = Y_train *100#(get rid of the digit)
Y_test = Y_test*100.
Y_test = Y_test.to_numpy()
#Random forest 
num_trees = 100
model3_accuracy_all = np.zeros((400))
RMSE3_all = np.zeros((400))

for i in range(len(Y_train.columns)):
    clf = RandomForestClassifier(n_estimators=num_trees).fit(X_train,Y_train.iloc[:,i]) #bagging numTrees trees for every of the 400 movies
    Y_predictions = clf.predict(X_test) 
    model3_accuracy_all[i] = accuracy_score(Y_test[:,i], Y_predictions)
    RMSE3_all[i] = sqrt(mean_squared_error(Y_test[:,i], Y_predictions)) 
#plot the first decision tree
fig = plt.figure(figsize=(150, 100))
plot_tree(clf.estimators_[0],fontsize=15,filled=True, impurity=True, 
          rounded=True)
#calculate average accuracy score 
average_accuracy3 = sum(model3_accuracy_all)/len(Y_train.columns)
print(average_accuracy3)
average_RMSE3 =  sum(RMSE2_all)/len(Y_train.columns)
print(average_RMS2)
##test: model for the first movie 

plt.plot(model3_accuracy_all,'o',markersize=3)
plt.xlabel('model of specific movie')
plt.ylabel('accuracy score')
plt.show() 

plt.plot(RMSE3_all,'o',markersize=3, color= 'red')
plt.xlabel('model of specific movie')
plt.ylabel('RMSE')
plt.show() 

x = sens
