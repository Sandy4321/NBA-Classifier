import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF

train = pd.read_csv('Users/jamesledoux/NBA_Classifier/train.csv')
test = pd.read_csv('Users/jamesledoux/NBA_Classifier/validation.csv')

test = test.dropna()
test = test.drop('Tm', axis=1)
test = test.drop('Player', axis=1)
test_answers = test['Pos']
test = test.drop('Pos', axis=1)
test = test.dropna()

train = train.drop('Tm', axis=1)
train = train.drop('Player', axis=1)
train = train.dropna()

#train = train[np.isfinite(train)]


x = train
x = train.drop('Pos', axis = 1)


test_answers = test_answers.replace('PG', 1)
test_answers = test_answers.replace('C', 5)
test_answers = test_answers.replace('PF', 4)
test_answers = test_answers.replace('SF', 3)
test_answers = test_answers.replace('SG', 2)
test_answers = test_answers.replace('SF-SG', 2.5)
test_answers = test_answers.replace('SG-SF', 2.5)
test_answers = test_answers.replace('PG-SG', 1.5)
test_answers = test_answers.replace('SG-PG', 1.5)
test_answers = test_answers.replace('SF-PF', 3.5)
test_answers = test_answers.replace('PF-SF', 3.5)
test_answers = test_answers.replace('PF-C', 4.5)
test_answers = test_answers.replace('C-PF', 4.5)
test_answers = test_answers.replace('C-SF', 4)
test_answers = test_answers.dropna()

train['Pos']

train['Pos'] = train['Pos'].replace('PG', 1)
train['Pos'] = train['Pos'].replace('C', 5)
train['Pos'] = train['Pos'].replace('PF', 4)
train['Pos'] = train['Pos'].replace('SF', 3)
train['Pos'] = train['Pos'].replace('SG', 2)
train['Pos'] = train['Pos'].replace('SF-SG', 2.5)
train['Pos'] = train['Pos'].replace('C-PF', 4.5)
train['Pos'] = train['Pos'].replace('PF-SF', 3.5)
train['Pos'] = train['Pos'].replace('PG-SG', 1.5)
train['Pos'] = train['Pos'].replace('SG-PF', 3)
train['Pos'] = train['Pos'].replace('SF-PF', 3.5)
train['Pos'] = train['Pos'].replace('SG-SF', 2.5)
train['Pos'] = train['Pos'].replace('PF-C', 4.5)
train['Pos'] = train['Pos'].replace('SG-PG', 1.5)
train = train.dropna()

y = train['Pos']

test = test.dropna()

import matplotlib.pyplot as plt
#show accuracy over various numbers of trees
acc = []
for t in range(2, 100):
    model = RF(n_estimators = t)  #orig set to 10
    model.fit(x,y)
    predicted_results = model.predict(test)
    for i in range(len(predicted_results)):   #round to nearest half position
        predicted_results[i] = round(predicted_results[i]*2)/2
    test_answers2 = []
    for i in test_answers:
        test_answers2.append(i)

    differences = []
    for i in range(len(test_answers2)):
        dif = predicted_results[i] - test_answers2[i]
        differences.append(dif)

    count = 0
    for i in differences:
        #if(i <= 0.5) and (i >=-0.5):
        if i == 0:
            count = count + 1
    acc.append(count)
    test_answers2 = []
    differences = []
plt.plot(acc)
#47% got EXACT position, 81% got it within a half position





train = pd.read_csv('train.csv')
test = pd.read_csv('validation.csv')

test = test.dropna()
test = test.drop('Tm', axis=1)
test = test.drop('Player', axis=1)

test = test[test['Pos'] != 'SG-SF']
test = test[test['Pos'] != 'SF-SG']
test = test[test['Pos'] != 'PG-SG']
test = test[test['Pos'] != 'SG-PG']
test = test[test['Pos'] != 'PF-SF']
test = test[test['Pos'] != 'PF-C']
test = test[test['Pos'] != 'SF-PF']
test = test[test['Pos'] != 'C-PF']

test_answers = test['Pos']
test = test.drop('Pos', axis=1)
test = test.dropna()

train = train.drop('Tm', axis=1)
train = train.drop('Player', axis=1)
train = train.dropna()
train = train[train['Pos'] != 'SG-SF']
train = train[train['Pos'] != 'SF-SG']
train = train[train['Pos'] != 'PG-SG']
train = train[train['Pos'] != 'SG-PG']
train = train[train['Pos'] != 'PF-SF']
train = train[train['Pos'] != 'PF-C']
train = train[train['Pos'] != 'SF-PF']
train = train[train['Pos'] != 'C-PF']







#df[df['Correlation'] >= 0]

#train = train[np.isfinite(train)]


x = train
x = train.drop('Pos', axis = 1)


test_answers = test_answers.replace('PG', 1)
test_answers = test_answers.replace('C', 5)
test_answers = test_answers.replace('PF', 4)
test_answers = test_answers.replace('SF', 3)
test_answers = test_answers.replace('SG', 2)
test_answers = test_answers.replace('SF-SG', 2.5)
test_answers = test_answers.replace('SG-SF', 2.5)
test_answers = test_answers.replace('PG-SG', 1.5)
test_answers = test_answers.replace('SG-PG', 1.5)
test_answers = test_answers.replace('SF-PF', 3.5)
test_answers = test_answers.replace('PF-SF', 3.5)
test_answers = test_answers.replace('PF-C', 4.5)
test_answers = test_answers.replace('C-PF', 4.5)
test_answers = test_answers.replace('C-SF', 4)
test_answers = test_answers.dropna()

train['Pos']

train['Pos'] = train['Pos'].replace('PG', 1)
train['Pos'] = train['Pos'].replace('C', 5)
train['Pos'] = train['Pos'].replace('PF', 4)
train['Pos'] = train['Pos'].replace('SF', 3)
train['Pos'] = train['Pos'].replace('SG', 2)
train['Pos'] = train['Pos'].replace('SF-SG', 2.5)
train['Pos'] = train['Pos'].replace('C-PF', 4.5)
train['Pos'] = train['Pos'].replace('PF-SF', 3.5)
train['Pos'] = train['Pos'].replace('PG-SG', 1.5)
train['Pos'] = train['Pos'].replace('SG-PF', 3)
train['Pos'] = train['Pos'].replace('SF-PF', 3.5)
train['Pos'] = train['Pos'].replace('SG-SF', 2.5)
train['Pos'] = train['Pos'].replace('PF-C', 4.5)
train['Pos'] = train['Pos'].replace('SG-PG', 1.5)
train = train.dropna()

y = train['Pos']

test = test.dropna()
model = RF(n_estimators = 10)
model.fit(x,y)
predicted_results = model.predict(test)

#round to nearest half position
for i in range(len(predicted_results)):
    predicted_results[i] = round(predicted_results[i])

test_answers2 = []
for i in test_answers:
    test_answers2.append(i)

differences = []
for i in range(len(test_answers2)):
    dif = predicted_results[i] - test_answers2[i]
    differences.append(dif)

count = 0
for i in differences:
    if(i == 0):
        count += 1
#2/3 accuracy based on pure positions alone 
        
        

#try changing 2/3 to 'wing' , and 4/5 to 'big'   only 1,2,3 categories. see what this does for accuracy
#create confusion matrix. cols actual positions, rows positions estimated. total freqs displayed.




















#train = train[np.isfinite(train)]

train = pd.read_csv('train.csv')
test = pd.read_csv('validation.csv')

test = test.dropna()
test = test.drop('Tm', axis=1)
test = test.drop('Player', axis=1)
test = test[test['Pos'] != 'PG-SG']
test = test[test['Pos'] != 'SG-PG']
test = test[test['Pos'] != 'PF-SF']
test = test[test['Pos'] != 'SF-PF']


test_answers = test['Pos']
test = test.drop('Pos', axis=1)
test = test.dropna()

train = train.drop('Tm', axis=1)
train = train.drop('Player', axis=1)
train = train.dropna()

train = train[train['Pos'] != 'PG-SG']
train = train[train['Pos'] != 'SG-PG']
train = train[train['Pos'] != 'PF-SF']
train = train[train['Pos'] != 'SF-PF']




x = train
x = train.drop('Pos', axis = 1)


test_answers = test_answers.replace('PG', 1)
test_answers = test_answers.replace('C', 3)
test_answers = test_answers.replace('PF', 3)
test_answers = test_answers.replace('SF', 2)
test_answers = test_answers.replace('SG', 2)
test_answers = test_answers.replace('SF-SG', 2)
test_answers = test_answers.replace('SG-SF', 2)
test_answers = test_answers.replace('PG-SG', 1.5)
test_answers = test_answers.replace('SG-PG', 1.5)
test_answers = test_answers.replace('SF-PF', 2.5)
test_answers = test_answers.replace('PF-SF', 2.5)
test_answers = test_answers.replace('PF-C', 3)
test_answers = test_answers.replace('C-PF', 3)
test_answers = test_answers.replace('C-SF', 3)
test_answers = test_answers.dropna()

train['Pos']

train['Pos'] = train['Pos'].replace('PG', 1)
train['Pos'] = train['Pos'].replace('C', 3)
train['Pos'] = train['Pos'].replace('PF', 3)
train['Pos'] = train['Pos'].replace('SF', 2)
train['Pos'] = train['Pos'].replace('SG', 2)
train['Pos'] = train['Pos'].replace('SF-SG', 2)
train['Pos'] = train['Pos'].replace('C-PF', 3)
train['Pos'] = train['Pos'].replace('PF-SF', 2.5)
train['Pos'] = train['Pos'].replace('PG-SG', 1.5)
train['Pos'] = train['Pos'].replace('SG-PF', 2)
train['Pos'] = train['Pos'].replace('SF-PF', 2)
train['Pos'] = train['Pos'].replace('SG-SF', 2)
train['Pos'] = train['Pos'].replace('PF-C', 3)
train['Pos'] = train['Pos'].replace('SG-PG', 1)
train = train.dropna()

y = train['Pos']

test = test.dropna()
model = RF(n_estimators = 10)
model.fit(x,y)
predicted_results = model.predict(test)

#round to nearest half position
for i in range(len(predicted_results)):
    predicted_results[i] = round(predicted_results[i])

test_answers2 = []
for i in test_answers:
    test_answers2.append(i)

differences = []
for i in range(len(test_answers2)):
    dif = predicted_results[i] - test_answers2[i]
    differences.append(dif)

count = 0
for i in differences:
    if(i == 0):
        count += 1
#83% accurate based on point/wing/post positions







    
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(train)


n_samples, n_features = data.shape
n_digits = 5   #len(np.unique(digits.target))
labels = digits.target

sample_size = 300   #experiment with this 

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
      
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=5)
kmeans.fit(reduced_data)

    
    
   # Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show() 
    
    
    
    
    
    