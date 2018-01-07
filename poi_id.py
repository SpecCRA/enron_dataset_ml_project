
# coding: utf-8

# In[1]:


#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list is a list of my selected features
# all_features is a list for exploration

features_list = ['poi', 'bon_plus_expenses', 'exercised_stock_options',
                'total_payments']
knn_list = ['poi']
all_features = ['poi', 'salary', 'bonus', 'long_term_incentive',
                'deferred_income', 'expenses', 'total_payments',
                'exercised_stock_options', 'restricted_stock', 'other', 'to_messages',
                'email_address', 'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi', 'to_msg_ratio',
                'from_msg_ratio', 'bon_plus_expenses', 'bon_sal_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)


# In[3]:


# Create another working dataframe to make new features
df_new = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).copy()


# In[4]:


# The following are my created features

# from_msg_ratio is ratio messages received from poi to total messages received
df_new['to_msg_ratio'] = df_new.from_this_person_to_poi.divide(df_new.to_messages, axis = 'index')

# create to_msg_ratio by dividing from_this_person_to_poi from to_messages
df_new['from_msg_ratio'] = df_new.from_poi_to_this_person.divide(df_new.from_messages, axis = 'index')

# create a new feature by adding expenses and bonus together
df_new['bon_plus_expenses'] = df_new['bonus'].add(df_new['expenses'], axis = 'index')
# new feature of bonus to salary ratio

df_new['bon_sal_ratio'] = df_new['bonus'].divide(df_new['salary'], axis = 'index')
# new feature of bonus to expenses ratio


# In[5]:


# Check how many missing values are in each column
print(df_new.isnull().sum())


# In[6]:


# Fill NaN with 0 where operations created NaN in some rows
df_new.fillna(0, inplace = True)


# In[7]:


# after you create features, the column names will be your new features
# create a list of column names:
new_features_list = df_new.columns.values
print(new_features_list)


# In[8]:


### Task 2: Remove outliers
# plot salary vs bonus as first step of outlier detection, visually
# uncomment the next line if using Jupyter Notebook for an inline plot
#%matplotlib inline
x = df_new['salary']
y = df_new['bonus']
plt.figure(figsize = (10, 8))
plt.scatter(x, y)
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()


# In[9]:


print("Highest bonus value: " + str(df_new['bonus'].max()))
print("Highest salary value: " + str(df_new['salary'].max()))


# In[10]:


# Identify the highest bonus and salary values to see what is going on
df_new.sort_values(['bonus', 'salary'], ascending=False).head()


# In[11]:


# Removed row "TOTAL" because it's not a proper data point, as in it's not an employee
df_new.drop(['TOTAL'], inplace=True)


# In[12]:


# Find how many POIs are left in the data
print("Number of POI in data set: " + str(len(df_new[(df_new['poi'] == True)])))


# In[13]:


# create a dictionary from the dataframe
df_dict = df_new.to_dict('index')


# In[14]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = df_dict


# In[15]:


# Check how many data points are left in my data
print("Number of data points: " + str(len(my_dataset)))


# In[16]:


### Extract features and labels from dataset for local testing
# Created one function for exploration then another for use after feature selection
exploration_data = featureFormat(my_dataset, all_features, sort_keys = True)
labels_exploration, features_exploration = targetFeatureSplit(exploration_data)

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[17]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


# In[18]:


selection = SelectKBest(k = 3)
selection.fit(features_exploration, labels_exploration)
print(selection.scores_)


# In[19]:


# Pipeline with KNearestNeighbors, first scaling with StandardScaler
# PCA is to help with KNN performance

knn = make_pipeline(StandardScaler(with_std = True),
                    SelectKBest(),
                    KNeighborsClassifier())
knn.fit(features_exploration, labels_exploration)


# In[20]:


# First one tried is RandomForestClassifier
rfc_exploration = RandomForestClassifier()
rfc_exploration = rfc_exploration.fit(features_exploration, labels_exploration)


# In[21]:


# Also trying a decision tree classifier because tree classifiers make sense here
dc_exploration = DecisionTreeClassifier()
dc_exploration= dc_exploration.fit(features_exploration, labels_exploration)


# In[22]:


# This function appends the feature and according importance value from tree
# classifier to a list to view more neatly
rfc_impt = []
dc_impt = []
selection_scores = []

def input_impt(impt_list, features_list, impts):
    for i in range(len(impts)):
        impt_list.append( (features_list[i], impts[i]) )

    impt_list.sort(key = lambda tup: tup[1], reverse = True)

    return impt_list


# In[23]:


# Call previous function to append and sort feature importances
input_impt(rfc_impt, all_features[1:], rfc_exploration.feature_importances_)
input_impt(dc_impt, all_features[1:], dc_exploration.feature_importances_)
input_impt(selection_scores, all_features[1:], selection.scores_)


# In[24]:


print("RandomForestClassifier importances values: ")
for item in rfc_impt:
    print(item[0] + " : " + str(item[1]))


# In[25]:


print("DecisionTreeClassifier importances values: ")
for item in dc_impt:
    print(item[0] + " : " + str(item[1]))


# In[26]:


#print "SelectKBest scores: "
#for item in selection_scores:
#    print item[0] + " : " + str(item[1])


# In[27]:


# Assign to new classifiers after choosing features

rfc = rfc_exploration.fit(features, labels)
dc = dc_exploration.fit(features, labels)


# In[28]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[29]:


# straified cv for parameters, 100 fold, and shuffled
best_cv = StratifiedShuffleSplit(n_splits = 100, random_state=42)


# In[30]:


# Function for entering classifiers into GridSearchCV with respective param_grids
# The output is time elapsed to run GridSearchCV and best_params_

def gridcv(clf, param_grid, cv, n_jobs, scoring):
    start_time = time.time()
    grid_cv = GridSearchCV(estimator = clf, param_grid = param_grid, cv = cv,
                          n_jobs = n_jobs, scoring = scoring)
    grid_cv.fit(features, labels)
    end_time = time.time()
    print("Minutes elapsed: " + str((float(end_time - start_time) / 60)))
    print(grid_cv.best_params_)


# In[31]:


# Parameter grid for RandomForestClassifier
# random_state parameter is to maintain consistency in output.

rfc_param_grid = {'n_estimators': [1,2, 3, 10, 100],
                 'min_samples_split': [2, 3, 5],
                'random_state': [42],
                 'max_features': [1, 2, 3],
                 'max_depth' : [2, 3, 5, 10, 50],
                 'min_samples_leaf': [1, 2, 3, 10]
                 }


# In[32]:


# gridsearchcv parameter grid for decisiontreeclassifier
# The list comprehension for max_features is just to make the feature selection
# process easier on me.
dc_param_grid = {'min_samples_split' : [2, 3, 4, 5, 10, 50],
                 'max_features' : [x for x in range(1, len(features_list))],
                 'min_samples_leaf': [1, 2, 3, 10, 20],
                'random_state' : [42]
                }


# In[33]:


# gridsearchcv parameter grid for KNeighborsClassifier
knn_param_grid = {'kneighborsclassifier__n_neighbors': [x for x in range(1, len(features_list))],
                  'kneighborsclassifier__algorithm': ['auto'],
                  'kneighborsclassifier__p': [1, 2],
                  'kneighborsclassifier__weights': ['uniform', 'distance'],
                  'selectkbest__k': [x for x in range(1, len(features_list))]
                 }


# In[34]:


# It took 94 minutes for me to run GridSearchCV for RandomForestClassifier.
#gridcv(rfc, rfc_param_grid, best_cv, 5, 'f1')


# In[35]:


# Assign clf to classifer chosen after testing with tester.py
# Parameters are selected from GridSearchCV's best_params_ attribute

#clf = RandomForestClassifier(min_samples_split = 2, n_estimators = 3,
#                            random_state = 42, max_depth = 50, min_samples_leaf = 1,
#                            max_features = 3)
#clf.fit(features, labels)


# In[36]:


gridcv(dc, dc_param_grid, best_cv, 5, 'f1')


# In[37]:


# Parameters are selected from GridSearchCV's best_params_ attributes
# I ended up choosing DecisionTreeClassifier because it performed better with
# precision and recall in tester.py
clf = DecisionTreeClassifier(min_samples_split = 2, random_state = 42,
                            max_features = 2, min_samples_leaf = 1)
clf.fit(features, labels)


# In[38]:


# Run GridSearchCV for KNeighborsClassifier parameters
gridcv(knn, knn_param_grid, best_cv, 5, 'f1')


# In[39]:


# Assign KNeighborsClassifier to clf for testing

#clf = make_pipeline(StandardScaler(with_std = True),
#                    SelectKBest(k = 3),
#                    KNeighborsClassifier(n_neighbors = 1, algorithm = 'auto',
#                                        weights = 'uniform',p = 2))
#clf.fit(features_exploration, labels_exploration)


# In[40]:


# This little bit of code is a quick preliminary check before running tester.py
labels_pred = clf.predict(features_test)
f1_score(labels_test, labels_pred)


# In[41]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

