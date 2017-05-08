import pandas as pd
import numpy as np
import copy,fbpca,yaml,hashlib,pickle,nltk,sys,nltk,optparse

parser = optparse.OptionParser()

parser.add_option("-r","--restrict",dest="restrict",help="restrict results to homogenous/heterogenous")
parser.add_option("-c","--cutoff",type="float",nargs=1,dest="cutoff", metavar="CUTOFF",help="save and display items whose quality index is greater than CUTOFF")
parser.add_option("-l","--limit",dest="limit",type="int",nargs=1,metavar="LIMIT",help="limit the number of results (randomly) to LIMIT, for all group sizes")

fields = yaml.load(open('wals-fields.yml'))
feature2field = dict()
for field,codes in fields.items():
    for code in codes:
        feature2field[code] = field

wals = pd.read_csv('language.csv',na_filter=False)
binarized = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
binarized = binarized.replace(to_replace="",value=0)

def locate_columns(minrows,numcols,cache,restrict=None,limit=None):
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
        i = None
        prev = locate_columns(minrows,numcols - 1,cache,restrict,limit)
        if limit is not None and len(prev) > 0:
            inds = np.random.choice(np.arange(len(prev)),min(len(prev),limit))
            i = 0
        for colgroup in prev:
            if isinstance(i,int):
                if i in inds:
                    continue 
                i += 1
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
                        if restrict is not None:
                            fields = fields_dict(newgroup)
                            if restrict == "heterogenous" and numcols > 4:
                                if fields.freq(fields.max()) > 0.8:
                                    continue
                            elif restrict == "homogenous":
                                if fields.B() > 1:
                                    continue
                            elif numcols > 4:
                                print("bad restriction:",restrict)
                            del(fields)
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

def locate_all_columns(minrows,restrict=None,limit=None):
    ret = list()
    for i in reversed(range(1,len(binarized.columns))):
        if len(locate_columns(minrows,i,ret,restrict,limit)) == 0:
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
        indicator = numvars*y[0]/total_inertia
        return indicator,y[0]/total_inertia,y[1]/total_inertia
    except Exception as e:
        print(str(e))        
        return None

def fields_dict(features):
    fs = dict()
    for f in features:
        code = f.split(" ")[0]
        field = feature2field[code]
        if field in fs:
            fs[field].append(code)
        else:
            fs[field] = [code]
    for f,l in fs.items():
        fs[f] = len(l)
    return nltk.FreqDist(fs)

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
features:
{6:s}""".format(
            self.numcols, 
            self.numrows, 
            self.quality_index, 
            self.dim1,
            self.dim2,
            self.fields.pformat().replace("FreqDist({","").strip("})"),
            "\n\r".join(self.cols)
        )
    
    def sort_fields(self):
        self.fields = fields_dict(self.cols)
 
if __name__ == '__main__':
    opts,args = parser.parse_args()
    flagged = list()
    asessed = set()
    qs = dict()
    n = int(args[0])
    qcutoff = float(opts.cutoff or 2)
    allcols = locate_all_columns(n,opts.restrict,opts.limit)
    for i in range(2,len(allcols)):
        icols = allcols[i]
        for cols in icols:
            groupkey = hashlib.md5("".join(sorted(cols)).encode('ascii')).digest()
            if groupkey not in asessed:
                asessment = asess(chunk_wals(cols))
                if asessment is not None:
                    qind = asessment[0]
                    k = str(i+1)+'-long'
                    if k  in qs:
                        qs[k].append(qind)
                    else:
                        qs[k] = [qind]
                    if  qind > qcutoff:
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
