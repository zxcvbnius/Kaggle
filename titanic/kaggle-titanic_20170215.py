#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:56:45 2017
@author: zxcvbnius
@title: Titanic (https://www.kaggle.com/c/titanic)

@desc: I am a newbie here. It's my first Kaggle project. Reference: [A Journey through Titanic
](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)
"""

# Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
# preview the data
print( titanic_df.head() )
print('-------------------')
print( titanic_df.info() )
print('-------------------')
print( test_df.info() )





# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)





# Embarked

# Only in titanic_df, fill the two missing values with the most occurred value, which is 'S'
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

# Virtualize - plot
sns.factorplot('Embarked', 'Survived', data=titanic_df, aspect=1.5, size=4)
fig, ([axis1, axis2, axis3, axis4]) = plt.subplots(1, 4, figsize=(15, 5))
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', data=titanic_df, ax=axis2)
sns.countplot(x='Survived', hue='Embarked', data=titanic_df, ax=axis3)


# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
print( embark_perc )
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis4)


# Either to consider Embarked column in predictions
# and remove 'S' dummy variable,
# and leave 'C'& 'Q' since they seem to have a good rate for Survival
embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)


# Or, don't create dummy variables for Embarked column, just drop it,
# because logically, Embarked doesn't seem to be useful in prediction.
titanic_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)




# Fare

# Only for test_df, since there is a missing 'Fare' value
# Using the median of fare to imputer missing value
test_df['Fare'].fillna( test_df['Fare'].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers
fare_not_survived = titanic_df['Fare'][titanic_df['Survived'] == 0]
fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]



# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])


# plot
fig, (axis5) = plt.subplots(1, 1, figsize=(20, 5))
sns.countplot(x='Fare',  data=titanic_df, ax=axis5)
# titanic_df['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim=(0, 50))

avgerage_fare.index.names = std_fare.index.names = ['Survived']
avgerage_fare.plot(yerr=std_fare, kind='bar', legend=False)





# Age


fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# get average, std, and number of NaN values in titanic_df
average_age_titanic = titanic_df['Age'].mean()
std_age_titanic = titanic_df['Age'].std()
count_nan_age_titanic = titanic_df['Age'].isnull().sum()

# get average, std, number of NaN values in test_df
average_age_test = test_df['Age'].mean()
std_age_test = test_df['Age'].std()
count_nan_age_test = test_df['Age'].isnull().sum()

# generate random numbers between [mean - std, mead + std]
rand_1 = np.random.randint( average_age_titanic - std_age_titanic , 
                           average_age_titanic + std_age_titanic,
                           size=count_nan_age_titanic)

rand_2 = np.random.randint( average_age_test - std_age_test , 
                           average_age_test + std_age_test,
                           size=count_nan_age_test)


# plot original Age values
# Note: drop all null values, and convert to int
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated
titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_1
test_df['Age'][np.isnan(test_df['Age'])] = rand_2


# covert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)


# plot new Age values
titanic_df['Age'].hist(bins=70, ax=axis2)

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue='Survived', aspect=2.5)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()


# average survived passengers by age
fig, axis1 = plt.subplots(1, 1, figsize=(15, 4))
average_age = titanic_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)




# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
titanic_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)




# Family

# Instead of having two columns Parch & SibSp
# We can have only on column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival of not
titanic_df['Family'] = titanic_df['Parch'] + titanic_df['SibSp']
titanic_df['Family'].loc[ titanic_df['Family'] > 0 ] = 1
titanic_df['Family'].loc[ titanic_df['Family'] == 0] = 0

test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'].loc[ test_df['Family'] > 0 ] = 1
test_df['Family'].loc[ test_df['Family'] == 0] = 0


# drop Parch & SibSp
titanic_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# virtualize
fig, (axis1 ,axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
sns.countplot(x='Family', data=titanic_df, order=[1, 0], ax=axis1)


# average of survived for those who had/didn't have any family member
family_perc = titanic_df[['Family','Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1, 0], ax=axis2)




# Sex

# As we wee, children (age < -16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)


# create dummy variables for Person column & drop Male as it has the lowest average of survived passengers
person_dummies_titanic = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Chid', 'Femal', 'Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)


person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Chid', 'Femal', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)


fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])

titanic_df.drop(['Sex', 'Person'], axis=1, inplace=True)
test_df.drop(['Sex', 'Person'], axis=1, inplace=True)




# Pclass

sns.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=titanic_df, size=5)

# create dummy variables for Pclass column, 
# and drop 3rd class at it has the lowest average of survived passengers
plcass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
plcass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
plcass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)


plcass_dummies_test = pd.get_dummies(test_df['Pclass'])
plcass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
plcass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)

titanic_df = titanic_df.join(plcass_dummies_titanic)
test_df = test_df.join(plcass_dummies_test)




# define training and testing data
X_train = titanic_df.drop(['Survived'], axis=1)
Y_train = titanic_df['Survived']
X_test  = test_df.drop('PassengerId', axis=1).copy()



# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print( 'Logisitc Regression : %s' % logreg.score(X_train, Y_train) )




# Support Vector Machines

"""
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print('SVC : %s ' % svc.score(X_train, Y_train) )
"""




# Random Forests

"""
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print('Random forest : %s ' % random_forest.score(X_train, Y_train) )
"""




# KNeighborsClassifier

"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print( 'KNN : %s ' % knn.score(X_train, Y_train) )
"""




# Gaussian Naive Bayes

"""
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
print( 'Gaussian Naive Bayes : %s ' % gaussian.score(X_train, Y_train) )
"""




# get Correlation Coefficeinet for each feature using Logisitc Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Coefficent Estimate'] = pd.Series(logreg.coef_[0])
print(coeff_df)




# preview
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred})
submission.to_csv('titanic.csv', index=False)