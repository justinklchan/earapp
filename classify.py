from sklearn import svm
from scipy import interp
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from operator import add
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

sns.set()
xrange=np.arange(2600)
phone="joint"
d0="/Users/justinkwoklamchan/data/"+phone+"/"

xall=np.load(d0+"xall.npy")
yall=np.load(d0+"yall.npy")
numcats=max(yall)+1
label=["pos","neg","err"]

def genCmat(preds,ytest):
    cmat={}
    for i in range(numcats):
        cmat[i]=[0]*numcats
    for i in range(len(preds)):
        p=preds[i]
        a=ytest[i]
        cmat[a][p]+=1
    for i in cmat:
        s=float(sum(cmat[i]))
        for j in range(len(cmat[i])):
            cmat[i][j]=cmat[i][j]/s
    return cmat

def printCmat(cmat):
    counter=0
    for i in cmat:
        sys.stdout.write(label[counter])
        for j in cmat[i]:
            sys.stdout.write("%10.2f "%j)
        counter+=1
        print

def avCmats(cmats):
    cmat=cmats[0]
    for i in range(1,len(cmats)):
        for k in cmat:
            cmat[k]=list(map(add,cmat[k],cmats[i][k]))
    
    for k in cmat:
        cmat[k]=[i/float(len(cmats)) for i in cmat[k]]
    return cmat

def cv_roc(clf):
    plt.figure()
    accs=[]
    cmats=[]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    kf = StratifiedKFold(n_splits=10)
    i = 0
    for train_index, test_index in kf.split(xall,yall):
        xtrain, xtest = xall[train_index], xall[test_index]
        ytrain, ytest = yall[train_index], yall[test_index]
        
        probas_=clf.fit(xtrain,ytrain).predict_proba(xtest)
        preds=clf.predict(xtest)
        acc=float(np.count_nonzero(preds==ytest))/len(ytest)
        accs.append(acc)
        cmat=genCmat(preds,ytest)
        cmats.append(cmat)
        
        fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i+=1
        print i,acc
    print "mean acc "+str(np.mean(accs))
    print
    printCmat(avCmats(cmats))
         
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    plt.show()
    
def cv(clf):
#     plt.figure()
    accs=[]
    cmats=[]
    kf = StratifiedKFold(n_splits=10)
    misclassifieds=np.array([])
    its=0
    nm=0
    for train_index, test_index in kf.split(xall,yall):
        xtrain, xtest = xall[train_index], xall[test_index]
        ytrain, ytest = yall[train_index], yall[test_index]
        clf.fit(xtrain,ytrain)
        
        preds=clf.predict(xtest)
        acc=float(np.count_nonzero(preds==ytest))/len(ytest)

#         for i in range(len(preds)):
#             if ytest[i]==0 and preds[i]==1:
#                 if misclassifieds.size==0:
#                     misclassifieds=xtest[i]
#                 else:
#                     misclassifieds=np.concatenate((misclassifieds,xtest[i]))
#                 nm+=1

        cmat=genCmat(preds,ytest)
        cmats.append(cmat)
        print its,acc
        accs.append(acc)
        its+=1
        
    print "final acc "+str(np.mean(accs))+"+/-"+str(np.std(accs))
    
    printCmat(avCmats(cmats))
    
#     misclassifieds=np.reshape(misclassifieds,(nm,2600))
#     np.random.shuffle(misclassifieds)
#     numlines=min(10,nm)
#     for i in range(numlines):
#         plt.plot(xrange,misclassifieds[i])
#     plt.ylim([0,1])
#     plt.show()

pca = PCA(20)
xall = pca.fit_transform(xall)
print np.sum(pca.explained_variance_ratio_)

# cv_roc(svm.SVC(probability=True))
# cv(svm.LinearSVC(penalty='l1',dual=False,C=1e3))
# cv(linear_model.LogisticRegression(penalty='l2'))

cv(BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5))
# cv(RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0))
# cv(ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0))
# cv(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0))
# cv(AdaBoostClassifier(n_estimators=100))
# cv(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0))
# cv(SGDClassifier(loss="hinge", penalty="l1"))
