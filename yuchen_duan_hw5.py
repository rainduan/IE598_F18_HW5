# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:38:23 2018

@author: duany
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression


df=pd.read_csv('wine.csv')
df.describe()
df.info()
df.head()

df.shape

cols = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline','Class']

for i in cols:
    plt.figure()
    sns.boxplot(x=i,data=df)
    plt.savefig('images/10_02.png', dpi=300)


sns.pairplot(df[cols], size=2)
plt.tight_layout()
plt.savefig('images/10_03.png', dpi=300)
plt.show()


cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=0.3)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 6},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.savefig('images/10_04.png', dpi=800)
plt.show()


X = df[['Flavanoids']].values
y = df['Class'].values
#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=20)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

# ## Estimating the coefficient of a regression model via scikit-learn


slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_train)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
# Print R^2 
print('R^2: %.3f' % slr.score(X_train, y_train))



lin_regplot(X_train, y_train, slr)
plt.xlabel('[Flavanoids] ')
plt.ylabel('[Class] ')

plt.savefig('images/10_07.png', dpi=300)
plt.show()

## end of EDA









# Splitting the data into 70% training and 30% test subsets.




X = df.iloc[:, :-1].values
y = df['Class'].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)


# Standardizing the data.




sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


lr = LogisticRegression()

#lr=svm.SVC()
lr = lr.fit(X_train_std, y_train)
print('lr train R^2: %.3f' % lr.score(X_train_std, y_train))
print('lr test R^2: %.3f' % lr.score(X_test_std, y_test))


sv=svm.SVC()

sv = sv.fit(X_train_std, y_train)

print('sv train R^2: %.3f' % sv.score(X_train_std, y_train))
print('sv test R^2: %.3f' % sv.score(X_test_std, y_test))
sv=svm.LinearSVC()


sv = sv.fit(X_train_std, y_train)

print('lsv train R^2: %.3f' % sv.score(X_train_std, y_train))
print('lsv test R^2: %.3f\n' % sv.score(X_test_std, y_test))

##baseline end



def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

##start of lr


lr = LogisticRegression()

#lr=svm.SVC()
lr = lr.fit(X_train_pca, y_train)




plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_04_00.png', dpi=300)
plt.show()
print('pca lr train R^2: %.3f' % lr.score(X_train_pca, y_train))

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_04_01.png', dpi=300)
plt.show()
print('pca lr test R^2: %.3f' % lr.score(X_test_pca, y_test))
##end of lr






##begin of sv



sv=svm.SVC()
#sv=svm.LinearSVC()

sv = sv.fit(X_train_pca, y_train)


plot_decision_regions(X_train_pca, y_train, classifier=sv)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_04_10.png', dpi=300)
plt.show()
print('pca sv train R^2: %.3f' % sv.score(X_train_pca, y_train))

plot_decision_regions(X_test_pca, y_test, classifier=sv)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_04_11.png', dpi=300)
plt.show()
print('pca sv test R^2: %.3f' % sv.score(X_test_pca, y_test))

##end of sv

##begin of LDA 

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)




##start of lr


lr = LogisticRegression()
#lr=svm.SVC()
lr = lr.fit(X_train_lda, y_train)




plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_05_00.png', dpi=300)
plt.show()
print('lda lr trainR^2: %.3f' % lr.score(X_train_lda, y_train))

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_05_01.png', dpi=300)
plt.show()
print('lda lr test R^2: %.3f' % lr.score(X_test_lda, y_test))
##end of lr






##begin of sv



sv=svm.SVC()
#sv=svm.LinearSVC()

sv = sv.fit(X_train_lda, y_train)


plot_decision_regions(X_train_lda, y_train, classifier=sv)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_05_10.png', dpi=300)
plt.show()
print('lda sv train R^2: %.3f' % sv.score(X_train_lda, y_train))

plot_decision_regions(X_test_lda, y_test, classifier=sv)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_05_11.png', dpi=300)
plt.show()
print('lda sv test R^2: %.3f' % sv.score(X_test_lda, y_test))

##end of sv

##end of lda

##begin kpca
gamma_space = np.logspace(-2, 1, 5)
for i in gamma_space:
    kpca = KernelPCA(n_components=2,kernel='rbf',gamma=i)
    X_train_kpca = kpca.fit_transform(X_train_std, y_train)
    X_test_kpca = kpca.transform(X_test_std)




##start of lr


    lr = LogisticRegression()
#lr=svm.SVC()
    lr = lr.fit(X_train_kpca, y_train)




    plot_decision_regions(X_train_kpca, y_train, classifier=lr)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('images/05_06_00gamma'+str(i)+'.png', dpi=300)
    plt.title('gamma='+str(i))
    plt.show()
    print('kpca lr trainR^2: %.3f' % lr.score(X_train_kpca, y_train))
    
    plot_decision_regions(X_test_kpca, y_test, classifier=lr)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('images/05_06_01gamma'+str(i)+'.png', dpi=300)
    plt.title('gamma='+str(i))
    plt.show()
    print('kpca lr test R^2: %.3f' % lr.score(X_test_kpca, y_test))
##end of lr






##begin of sv



    #sv=svm.SVC()
    sv=svm.LinearSVC()

    sv = sv.fit(X_train_kpca, y_train)


    plot_decision_regions(X_train_kpca, y_train, classifier=sv)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('images/05_06_10gamma'+str(i)+'.png', dpi=300)
    plt.title('gamma='+str(i))

    plt.show()
    print('kpca sv train R^2: %.3f' % sv.score(X_train_kpca, y_train))
    
    plot_decision_regions(X_test_kpca, y_test, classifier=sv)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('images/05_06_11gamma'+str(i)+'.png', dpi=300)
    plt.title('gamma='+str(i))

    plt.show()
    print('kpca sv test R^2: %.3f\n' % sv.score(X_test_kpca, y_test))

##end of sv



print("My name is Yuchen Duan")
print("My NetID is: yuchend3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################