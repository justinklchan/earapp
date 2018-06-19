indir="/Users/justinkwoklamchan/data/joint/err-fft"
outdir=indir+"-norm"
m={"controlBottom":"cb","bottomLeft":"lb","bottomRight":'rb'}

def getPid(fname):
    if "pulse" in fname:
        return int(fname.split("-")[1])
    else:
        return int(fname.split("p")[1].split("-")[0])
    
def getType(fname):
    if "pulse" in fname:
        return m[fname.split("-")[2]]
    else:
        return fname.split("-")[1]

from os import listdir
from os.path import isfile, join
s6 = [f for f in listdir(indir) if isfile(join(indir, f)) and f.endswith(".dat") and f.startswith("pulse")]
iphone = [f for f in listdir(indir) if isfile(join(indir, f)) and f.endswith(".dat") and not f.startswith("pulse")]
s6=sorted(s6)
iphone=sorted(iphone)

pids=[15,17,18,19,20,21,22,23,24,25,26,27]

def mult(a,b):
    c=[]
    for i in range(len(a)):
        c.append(a[i]*b[i])
    return c 

for pid in pids:
    print pid
    files=[i for i in iphone if getPid(i)==pid]
    if len(files)>0:
        l=[float(i) for i in open(indir+"/"+files[0]).read().split("\n")[:-1]]
        weights = [1/i if i!=0 else 0 for i in l]
        for x in range(1,len(files)):
            ar=[float(i) for i in open(indir+"/"+files[x]).read().split("\n")[:-1]]
            weighted=mult(weights,ar)
            outf=open(outdir+"/"+files[x],"w+")
            for w in weighted:
                outf.write(str(w)+"\n")
    






