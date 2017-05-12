import pandas as pd
import numpy as np
import copy,fbpca,yaml,hashlib,pickle,nltk,sys,nltk,optparse
from progress import progress

parser = optparse.OptionParser()

parser.add_option(
    "--heterogenous",
    type="float",
    nargs=1,
    metavar="MAX",
    dest="heterogenous",
    help="restrict feature fields to not more than MAX percent of the features in the group"
)

parser.add_option(
    "--homogenous",
    type="str",
    nargs=1,
    metavar="WALS_FIELD",
    dest="homogenous",
    help="select features only from the specified field (original WALS names, lower case, underscores instead of spaces"
)

parser.add_option(
    "-c",
    "--cutoff",
    type="float",
    nargs=1,
    dest="cutoff",
    metavar="CUTOFF",
    help="save and display items whose quality index is greater than CUTOFF"
)

parser.add_option(
    "-l",
    "--limit",
    dest="limit",
    type="float",
    nargs=1,
    metavar="LIMIT",
    help="proportion of possible candidates to prune at each iteration. for example if set to 1.0, then k feature groups of length n will spawn 192*k groups of length n + 1"
)

fields = yaml.load(open('wals-fields.yml'))
feature2field = dict()
for field,codes in fields.items():
    for code in codes:
        feature2field[code] = field

wals = pd.read_csv('language.csv',na_filter=False)
binarized = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
binarized = binarized.replace(to_replace="",value=0)
totalfeatures = len(binarized.columns)

def locate_columns(minrows,numcols,cache,heterogenous=None,limit=None):
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
        rnds = None
        prev = locate_columns(minrows,numcols - 1,cache,heterogenous,limit)
        willcheck = len(prev) * totalfeatures * (numcols - 1)
        # for efficient coin tossing down the road
        if limit is not None and numcols > 2:
            rnds = np.random.random_sample(willcheck)
        skipped = 0
        cur = 0
        if willcheck > 0:
            print("checking {0:.1f}K feature groups of length {1:d}".format(willcheck/1000,numcols))
        for colgroup in prev:

            for add in binarized.columns:
                if add in colgroup:
                    cur += numcols - 1
                    continue
                cand = True
                inthisg = 1
                for s in subs(colgroup):
                    # if the random restriction is on
                    # toss a coin here to decide whether to skip this or not
                    if rnds is not None and rnds[cur] > limit/numcols:
                        skipped += 1
                        cur += 1
                        cand = False
                        continue
                    if sorted(s + [add]) not in prev:
                        cur += numcols  - inthisg
                        cand = False
                        break
                    else:
                        cur += 1
                    inthisg += 1

                
                if cand:
                    newgroup = sorted(colgroup + [add])
                    if newgroup not in ret:
                        if heterogenous is not None and numcols > 4:
                            fields = fields_dict(newgroup)
                            if fields.freq(fields.max()) > heterogenous:
                                skipped += 1
                                continue
                            del(fields)
                        if satisfies(newgroup,minrows):
                            ret.append(newgroup)
            
            progress(cur,willcheck,50,"skipped:{:.1f}K".format(skipped/1000))
        if len(ret) > 0:
            print()
            cache.insert(numcols,ret)
    #print("exiting",minrows,numcols)
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

def locate_all_columns(minrows,heterogenous=None,limit=None):
    ret = list()
    for i in reversed(range(1,len(binarized.columns)+1)):
        if len(locate_columns(minrows,i,ret,heterogenous,limit)) == 0:
            break
    print()
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
        y = np.square(y)
        total_inertia = numcats/numvars - 1
        indicator = numcats*y[0]/total_inertia
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
    savefile = "feature-sets/colgroups-{:d}".format(n)
    if opts.homogenous in fields.keys():
        binarized = binarized[[c for c in binarized.columns if feature2field[c.split(" ")[0]] == opts.homogenous]]
        savefile += "-homogenous-{}".format(opts.homogenous)
        totalfeatures = len(binarized.columns)
    qcutoff = float(opts.cutoff or 2)
    allcols = locate_all_columns(n,opts.heterogenous,opts.limit)
    for i in range(2,len(allcols)):
        icols = allcols[i]
        for cols in icols:
            groupkey = hashlib.md5("".join(sorted(cols)).encode('ascii')).digest()
            if groupkey not in asessed:
                asessment = asess(chunk_wals(cols))
                if asessment is not None:
                    qind = asessment[0]
                    k = str(i+1)+'-long'
                    if k in qs:
                        qs[k].append(qind)
                    else:
                        qs[k] = [qind]
                    if  qind > qcutoff:
                        colgroup = ColGroup(cols,*asessment)
                        flagged.append(colgroup)
                        print("flagging:",colgroup)
                        print()
                asessed.add(groupkey)
    if opts.heterogenous:
        savefile += "-heterogenous-{:.1f}".format(opts.heterogenous)
    if opts.limit:
        savefile += "-random-limit-ratio-{:.1f}".format(opts.limit)
    if len(flagged) > 0:
        if not os.path.isdir('feature-sets'):
            os.mkdir('feature-sets')
        with open(savefile+'.pkl','wb') as f:
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
