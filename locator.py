import pandas as pd
import numpy as np
import copy,fbpca,yaml,hashlib,pickle,nltk,sys
fields = yaml.load(open('wals-fields.yml'))
feature2field = dict()
for field,codes in fields.items():
    for code in codes:
        feature2field[code] = field

wals = pd.read_csv('language.csv',na_filter=False)
binarized = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
binarized = binarized.replace(to_replace="",value=0)

def locate_columns(minrows,numcols,cache):
    print("entering",minrows,numcols)
    ret = list()
    if numcols == 1:
        cols = binarized.columns[binarized.sum(axis=0) >= minrows]
        ret = [[c] for c in cols]
        cache.append(ret)
    elif len(cache) >= numcols:
        ret =  cache[numcols - 1]
    else:
        ret = list()
        prev = locate_columns(minrows,numcols - 1,cache)
        for colgroup in prev:
            for add in binarized.columns:
                if add in colgroup:
                    continue
                cand = True
                for s in subs(colgroup):
                    if sorted(s + [add]) not in prev:
                        cand = False
                        break
                if cand:
                    newgroup = sorted(colgroup + [add])
                    if newgroup not in ret:
                        if satisfies(newgroup,minrows):
                            ret.append(newgroup)
        if len(ret) > 0:
            cache.insert(numcols,ret)
    print("exiting",minrows,numcols)
    return ret

def subs(l):
    for item in l:
        cop = copy.deepcopy(l)
        cop.remove(item)
        yield cop
        
def satisfies(colgroup,minrows):
    sums = np.sum(binarized[colgroup],axis=1)
    fullrows = len(sums[sums == len(colgroup)]) 
    return fullrows >= minrows

def locate_all_columns(minrows):
    ret = list()
    for i in reversed(range(1,len(binarized.columns))):
        if len(locate_columns(minrows,i,ret)) == 0:
            break
        else:
            print("finished locating all",minrows,i)
    return ret
        
def chunk_wals(columns,chunk=True,just_actives=True):
    bchunk = binarized[columns]
    indices = bchunk.sum(axis=1) == len(columns)
    if chunk:
        cols = columns if just_actives else np.concatenate((wals.columns[0:10],columns))
        return wals[indices][cols]
    else:
        return np.count_nonzero(indices)

def asess(df):
    numvars = len(df.columns)
    rawdata = pd.get_dummies(df).values
    numcats = rawdata.shape[1]
    N = np.sum(rawdata)
    P = rawdata/N
    csums = P.sum(axis=0)
    rsums = P.sum(axis=1)
    expected = rsums.reshape(-1,1).dot(csums.reshape(1,-1))
    stand = np.diag(1/np.sqrt(rsums)).dot(P - expected).dot(np.diag(1/np.sqrt(csums)))
    try:
        x,y,z = fbpca.pca(stand,numcats,raw=True)
        #trace = np.sum(np.var(rawdata,axis=0))
        #indicator = numcols * y[0] / trace
        y = np.square(y)
        total_inertia = np.sum(y)
        indicator = numvars*(y[0] + y[1])/total_inertia
        return indicator,y[0]/total_inertia,y[1]/total_inertia
    except Exception as e:
        print(str(e))        

class ColGroup:
    def __init__(self,cols,quality_index,dim1,dim2):
        self.cols = cols
        self.quality_index = quality_index
        self.dim1 = dim1
        self.dim2 = dim2
        self.numcols = len(cols)
        self.numrows = chunk_wals(self.cols,False)
        self.sort_fields()
    
    def __str__(self) :
        return """{0:d} long group covering {1:d} languages
quality index: {2:.2f}
dim1: {3:.0%}
dim2: {4:.0%}
fields: {5:s}
columns:
{6:s}""".format(
            self.numcols, 
            self.numrows, 
            self.quality_index, 
            self.dim1,
            self.dim2,
            self.fields_repr(),
            "\n\r".join(self.cols)
        )
    
    def sort_fields(self):
        fs = dict()
        for col in self.cols:
            code = col.split(" ")[0]
            field = feature2field[code]
            if field in fs:
                fs[field].append(code)
            else:
                fs[field] = [code]
        for f,l in fs.items():
            fs[f] = (l,len(l))
        self.fields = fs
            
    def fields_repr(self):
        ret = ""
        for f,(li,le) in self.fields.items():
            ret += f+":"+str(le)+" "
        return ret.strip()

    def field_proportions(self):
        ret = list()
        for f,(li,le) in self.fields.items():
            ret.append(le/self.numcols)
        return sorted(ret)
            
            


if __name__ == '__main__':
    #dat = chunk_wals(['86A Order of Genitive and Noun',
    #'87A Order of Adjective and Noun',
    #'90A Order of Relative Clause and Noun'
    #])
    flagged = list()
    asessed = set()
    qs = dict()
    n = int(sys.argv[1])
    allcols = locate_all_columns(n)
    for i in range(2,len(allcols)):
        icols = allcols[i]
        for cols in icols:
            groupkey = hashlib.md5("".join(sorted(cols)).encode('ascii')).digest()
            if groupkey not in asessed:
                asessment = asess(chunk_wals(cols))
                qind = asessment[0]
                k = str(i+1)+'-long'
                if k  in qs:
                    qs[k].append(qind)
                else:
                    qs[k] = [qind]
                if  qind > 1.5:
                    colgroup = ColGroup(cols,*asessment)
                    flagged.append(colgroup)
                    print("flagging:",colgroup)
                    print()
                asessed.add(groupkey)
    with open('flagged-colgroups-'+str(n)+'.pkl','wb') as f:
        pickle.dump(flagged,f)

    qindstats = list()
    for l in qs.values():
        qindstats.append({
            'max' : np.max(l),
            'min' : np.min(l),
            'mean': np.mean(l),
            '3rdq': np.percentile(l,0.75),
            'std' : np.sqrt(np.var(l))
        })
    qindstats = pd.DataFrame(qindstats,index=qs.keys())
