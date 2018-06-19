import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

phone="s6"
d0="/Users/justinkwoklamchan/data/"+phone+"/"
d=["pos-fft","neg-fft","err-fft"]
inoutear=False

def getLabels(ar,onehot=True):
    numclasses=len(ar)
    bigar=[]
    for i in range(numclasses):
        numinstances=ar[i]
        bigar.extend([i]*numinstances)
    a=np.asarray(bigar)
    if onehot:
        b = np.zeros((len(a), numclasses))
        b[np.arange(len(a)), a] = 1
        return b
    else:
        return a

def shuffle2(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def blacklist(files,dir):
    checkdir=dir[:3]+"-temp"
    checkpath=d0+checkdir
    fs=sorted([f for f in listdir(checkpath) if isfile(join(checkpath, f)) and "DS_Store" not in f])
    newfiles=[]
    for fi in files:
        isin=False
        for fj in fs:
            if fj[:-4] in fi:
                isin=True
                break
        if not isin:
            newfiles.append(fi)
    print len(newfiles),len(files)
    return newfiles

def limitchirps(fs,lim):
    newfs=[]
    for f in fs:
        if getChirpNum(f) <= lim:
           newfs.append(f)
    return newfs 

def getChirpNum(fname):
    if "pulse" in fname:
        elt=fname.split("-")[4]
        return int(elt[:elt.index(".")])
    else:
        elt=fname.split("-")[2]
        return int(elt[:elt.index(".")])

def getData():
    counts=[]
    x=np.array([])
    for di in d:
        path=d0+di
        fs = sorted([f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".dat")])
#         fs=blacklist(fs,di)
#         fs=limitchirps(fs,9)
        counts.append(len(fs))
        
        for fi in tqdm(fs):
            if x.size==0:
                x=np.loadtxt(path+"/"+fi)
            else:
                y=np.loadtxt(path+"/"+fi)
                x=np.concatenate((x,y))
                  
    x=np.reshape(x,(sum(counts),2600))
    
#     rescaling
    xmin=np.min(x)
    xmax=np.max(x)
    span=xmax-xmin
    x=(x-xmin)/span

    return x,counts

x,counts=getData()
if inoutear and len(d)==3:
    counts=[counts[d.index("pos-fft")]+counts[d.index("neg-fft")],counts[d.index("err-fft")]]
y=getLabels(counts,onehot=False)
    
n=len(x)

np.save(d0+"xall",x)
np.save(d0+"yall",y)

shuffle2(x,y)

def output1():
    train_val_split=int(.7*n)
    val_test_split=int(.9*n)
    
    xtrain=x[:train_val_split]
    ytrain=y[:train_val_split]
    xval=x[train_val_split:val_test_split]
    yval=y[train_val_split:val_test_split]
    xtest=x[val_test_split:]
    ytest=y[val_test_split:]
    
    np.save(d0+"xtrain",xtrain)
    np.save(d0+"ytrain",ytrain)
    np.save(d0+"xval",xval)
    np.save(d0+"yval",yval)
    np.save(d0+"xtest",xtest)
    np.save(d0+"ytest",ytest)
    
def output2():
    train_test_split=int(.8*n)
    xtrain=x[:train_test_split]
    ytrain=y[:train_test_split]
    xtest=x[train_test_split:]
    ytest=y[train_test_split:]
    np.save(d0+"xtrain",xtrain)
    np.save(d0+"ytrain",ytrain)
    np.save(d0+"xtest",xtest)
    np.save(d0+"ytest",ytest)

# output2()
print counts
print "done"