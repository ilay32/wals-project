import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as km
from sklearn.metrics import silhouette_score as silsc
from scipy.spatial.distance import hamming
import copy,fbpca,yaml,hashlib,pickle,nltk,sys,nltk,optparse,os,prince
from progress import progress
from matplotlib import pyplot as plt
from spectral import kmeans

parser = optparse.OptionParser()

parser.add_option(
    "-s",
    "--silhouette-score-cutoff",
    dest="sil_cutoff",
    metavar="SIL_CUTOFF",
    help="flag (and save) only results with silhoette score larger than SIL_CUTOFF in addition to the CUTOFF restriction."
)

parser.add_option(
    "-d",
    "--dry-run",
    dest="dryrun",
    metavar="DRYRUN",
    action="store_true",
    help="don't save flagged results to disk"
)

parser.add_option(
    "-a",
    "--allow_empty",
    dest="allow_empty",
    metavar="EMPTY",
    type="int",
    help="specify number of empty cells permitted for feature columns"
)
    
parser.add_option(
    "-w",
    "--with-clustering",
    action="store_true",
    dest="with_clustering",
    help="when set, the evaluation of a feature group will perform clustering of the two large families and check the resulting silhouette coefficient"
)

parser.add_option(
    "-e",
    "--exclude",
    action="append",
    type="string",
    metavar="EXCLUDE",
    dest="exclude",
    help="Exclude the specified field(s) (underscore, lower case...)"
)

parser.add_option(
    "-i",
    "--include",
    metavar="INCLUDE",
    action="append",
    type="string",
    dest="include",
    help="Include the specified fields(s) (underscore, lower case...)"
)

parser.add_option(
    "--heterogenous",
    type="float",
    nargs=1,
    metavar="MAX",
    dest="heterogenous",
    help="restrict feature fields to not more than MAX percent of the features in the group"
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

# globals #
#--------#
wals = pd.read_csv('wals.csv',na_filter=False)
binarized = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
binarized = binarized.replace(to_replace="",value=0)

headclass_indices = islang(wals,heads)
ratioclass_indices = islang(wals,ratios)

featfullnames = wals.columns[10:]
feature2area = dict()
code2feature = dict()
areas = yaml.load(open('wals-areas.yml'))
for area,codes in areas.items():
    for code in codes:
        feature2area[code] = area
for name in featfullnames:
    code = name.split(" ")[0]
    code2feature[code] = name
    wals = wals.rename(columns={name : code})
    binarized = binarized.rename(columns={name : code})

        


# helpers #
#---------#
def verify_areas(l):
    for f in l:
        if f not in areas.keys():
            print("include/exclude: arguments must be WALS feature areas lower-cased, underscore for space")
            quit()
    return True


def subs(l):
    for item in l:
        cop = copy.deepcopy(l)
        cop.remove(item)
        yield cop

def chunk_wals(columns,chunk=True,just_actives=True,allow_empty=0):
    bchunk = binarized[columns]
    full = bchunk.sum(axis=1) == len(columns)
    indices = bchunk.index[full]
    if allow_empty > 0:
        notfull = bchunk[~full]
        notfull = notfull[notfull.sum(axis=1) > len(columns) - 2]
        if len(notfull):
            add = notfull.sample(allow_empty)
            almostfull = bchunk[full].append(add).sort_index()
            indices = almostfull.index
    if chunk:
        cols = columns if just_actives else np.concatenate((wals.columns[0:10],columns))
        return wals.loc[indices][cols]
    else:
        return np.count_nonzero(full)

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


       
class Locator:
    def __init__(self,minrows,cutoff=2,sil_cutoff=0.5,limit=None,heterogenous=None, \
        include=None,exclude=None,with_clustering=False,allow_empty=0,dryrun=False,target_langs=None):
        self.minrows = minrows
        self.dryrun = dryrun
        self.cutoff = float(cutoff or 2)
        self.sil_cutoff = sil_cutoff
        self.with_clustering = with_clustering
        self.random_limit = limit
        self.heterogenous = heterogenous
        self.allow_empty = int(allow_empty or 0)
        self.hetreject = list()
        self.include = include
        self.exclude = exclude
        self.cache = list()
        self.limit = limit
        self.target_langs = target_langs
        self.totalfeatures = 0

    def satisfies(self,colgroup):
        l = len(colgroup)
        minrows = self.minrows
        sums = np.sum(binarized[colgroup],axis=1)
        subtract = self.allow_empty if l > self.allow_empty else 0
        fullrows = sums[sums >= (l - subtract)]
        if self.target_langs is not None:
            fullrows = fullrows[fullrows.index.intersection(self.target_langs)]
        # if the heterogeneity requirement is on, and the group is too homogenous,
        # check if there's a subset of the covered languages that's larger than minrows and does not
        # include features from the dominant field
        if self.heterogenous is not None and len(fullrows) > minrows:
            fields = nltk.FreqDist([feature2area[c] for c in colgroup])
            if fields.freq(fields.max()) > self.heterogenous:
                for feature in binarized.columns:
                    if feature2area[feature] != fields.max() and feature not in colgroup:
                        sums2 = np.sum(binarized[colgroup + [feature]],axis=1)
                        fullrows2 = len(sums2[sums2 == l + 1])
                        if fullrows2 > minrows:
                            return True
                # the loop completed so this group can't be extended with non-dominant fields
                if len(self.hetreject) > l:
                    self.hetreject[l] += 1
                else:
                    self.hetreject.append(1)
                return False
        return len(fullrows) > minrows



    def locate_columns(self,numcols):
        minrows = self.minrows
        print("entering",minrows,numcols)
        ret = list()
        if numcols == 1:
            cols = binarized.columns[binarized.sum(axis=0) >= minrows]
            ret = [[c] for c in cols]
            self.cache.append(ret)
        elif len(self.cache) >= numcols:
            ret =  self.cache[numcols - 1]
        else:
            ret = list()
            rnds = None
            prev = self.locate_columns(numcols - 1)
            willcheck = len(prev) * self.totalfeatures * (numcols - 1)
            
            # for efficient coin tossing down the road
            if self.limit is not None and numcols > 2:
                rnds = np.random.random_sample(willcheck)
            skipped = 0
            cur = 0
            passed = 0
            if willcheck > 0:
                print("checking {0:.1f}K feature groups of length {1:d}".format(willcheck/1000,numcols))
            for colgroup in prev:
                for add in binarized.columns:
                    if add in colgroup:
                        cur += numcols - 1
                        continue
                    cand = True
                    inthisg = 0
                    for s in subs(colgroup):
                        inthisg += 1
                        # if the random restriction is on
                        # toss a coin here to decide whether to skip this or not
                        if rnds is not None and rnds[cur] > float(self.limit)/numcols:
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

                    if cand:
                        newgroup = sorted(colgroup + [add])
                        if newgroup not in ret:
                            if self.satisfies(newgroup):
                                passed += 1
                                ret.append(newgroup)
                
                progress(cur,willcheck,50,"skipped:{0:.1f}K passed: {1:d}".format(skipped/1000, passed))
            if len(ret) > 0:
                print()
                self.cache.insert(numcols,ret)
        return ret

       
    def locate_all_columns(self):
        for i in reversed(range(1,len(binarized.columns)+1)):
            if len(self.locate_columns(i)) == 0:
                break
        print()
        return self.cache
            
    def main(self):
        global binarized
        flagged = list()
        asessed = set()
        qs = dict()
        savefile = os.path.join("feature-sets","colgroups-{:d}".format(self.minrows))
        if self.include is not None: 
            inc = self.include if isinstance(self.include,list) else [self.include]
            if verify_areas(inc): 
                binarized = binarized[[c for c in binarized.columns if feature2area[c] in inc]]
                savefile += "-inc-{}".format("-".join(inc))
        if self.exclude is not None:
            exc = self.exclude if isinstance(self.exclude,list) else [self.exclude]
            if verify_areas(exc):
                binarized = binarized[[c for c in binarized.columns if feature2area[c] not in exc]]
                savefile += "-exc-{}".format("-".join(exc))
        self.totalfeatures = len(binarized.columns)
        allcols = self.locate_all_columns()
        for i in range(2,len(allcols)):
            print("asessing {:d} long groups".format(i+1))
            icols = allcols[i]
            for j,cols in enumerate(icols):
                progress(j,len(icols),50,"flagged:{:d}".format(len(flagged)))
                groupkey = "".join(sorted(cols))
                if groupkey not in asessed:
                    asessment = asess(chunk_wals(cols))
                    if asessment is not None:
                        qind = asessment[0]
                        k = str(i+1)+'-long'
                        if k in qs:
                            qs[k].append(qind)
                        else:
                            qs[k] = [qind]
                        if  qind > self.cutoff:
                            colgroup = ColGroup(cols,asessment,allow_empty=self.allow_empty)
                            if self.with_clustering:
                                sil = colgroup.gen_separation()
                            if (self.with_clustering and sil >= self.sil_cutoff) or not self.with_clustering:
                                flagged.append(colgroup)
                    asessed.add(groupkey)
            print()

        if self.heterogenous:
            savefile += "-heterogenous-{:.1f}".format(self.heterogenous)
        if self.limit:
            savefile += "-randlimit-{:.1f}".format(self.limit)
        if len(flagged) > 0 and not self.dryrun:
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
        self.qindstats = pd.DataFrame(qindstats,index=qs.keys())
        self.flagged = flagged

class ColGroup:
    csv_dir = 'chunked-feature-sets'
    def __init__(self,cols,asessment=None,allow_empty=0):
        self.cols = cols
        self.colnames = [code2feature[c] for c in cols]
        self.asessment = asessment or asess(chunk_wals(cols))
        self.quality_index = asessment[0]
        self.dim1  = asessment[1]
        self.dim2 = asessment[2]
        self.numcols = len(cols)
        self.numrows = chunk_wals(self.cols,False)
        self.fields = nltk.FreqDist([feature2area[c] for c in self.cols])
        self.silhouette_score = None
        self.allow_empty = allow_empty
        self.mca = None
    
    def fields_spread(self):
        return self.fields.most_common(1)[0][0]/self.fields.N()
    
    def gen_separation(self,n_clusts=2):
        df =  chunk_wals(self.cols,True,False)
        families = nltk.FreqDist(df['family'])
        topfams = [fam[0] for fam in families.most_common(n_clusts)]
        active = df[[c for c in df.columns if c in binarized.columns]]
        mca = prince.MCA(active,n_components=-1)
        labels = df['family'][df['family'].isin(topfams)]
        silhouettes = list()
        binact = pd.get_dummies(active)
        filtered = binact.loc[labels.index]
        silhouettes.append((-1,silsc(filtered,labels)))
        silhouettes.append((0,silsc(filtered,labels,hamming)))
        for i in range(1,len(mca.eigenvalues)+1):
            filtered = mca.row_principal_coordinates[np.arange(i)].loc[labels.index]
            silhouettes.append((i,silsc(filtered,labels)))
        self.silhouettes = silhouettes
        self.silhouette_score = sorted(silhouettes,key=lambda x: x[1],reverse=True)[0][1]
        self.families = families
        self.mca = mca
        return self.silhouette_score
    
    def plot_silhouettes(self):
        thresh = 0.5
        maxdim = self.silhouettes[-1][0]
        minscore = sorted(self.silhouettes,key=lambda x: x[1])[0][1]
        sd = self.significant_dimensions(thresh)
        plt.ylabel('average silhouette coefficient')
        plt.xlabel('dimensions used')
        x = [d for d,s in self.silhouettes]
        y = [s for d,s in self.silhouettes]
        
        plt.scatter(x[0],y[0],color='g',label='without MCA (Eucledian)')
        plt.scatter(x[1],y[1],color='r',label='without MCA (Hamming)')
        
        plt.plot(x[2:],y[2:],label='by MCA projection')
        plt.xticks(np.arange(0,maxdim))
        plt.grid()
        plt.vlines([sd],ymin=minscore,ymax=self.silhouette_score,linestyles="dashed",label="{:.1f} of the variance explained".format(thresh))
        plt.legend()
    
    def add_clustering_data(self,n_clusts=2,raw=False):
        self.gen_separation(n_clusts)
        df =  chunk_wals(self.cols,True,False)
        dims = sorted(self.silhouettes,key=lambda x: x[1],reverse=True)[0][0]
        topfams = [fam[0] for fam in self.families.most_common(n_clusts)]
        labels = df['family'][df['family'].isin(topfams)]
        if raw:
            dims = 'raw'
            starts = list()
            active = pd.get_dummies(df[[c for c in df.columns if c in binarized.columns]])
            for f in topfams:
                choice = labels[labels == f].sample(1).index
                starts.append(active.loc[choice].values)
            filtered = np.array([[pd.get_dummies(active).loc[i].values] for i in labels.index])
            print(filtered)
            pred,means = kmeans(filtered,n_clusts,start_clusters=np.array(starts),distance='L1') 
            pred = pred.flatten()
            print(pred)
        else:
            filtered = self.mca.row_principal_coordinates[np.arange(dims)].loc[labels.index]
            pred = km(n_clusters=n_clusts,n_init=15).fit_predict(filtered)
        addcolumn = list()
        prediter = iter(pred)
        for i in df.index:
            addcolumn.append(next(prediter) if i in labels.index else 'other')
        df["kmpredict-{}-{}".format(n_clusts,dims)] = pd.Series(addcolumn,index=df.index)
        return df
        
    def significant_dimensions(self,thresh=0.6):
        for i,cm in enumerate(self.mca.cumulative_explained_inertia,1):
            if cm >= thresh:
                return i

    def to_csv(self,binary=False,allow_empty=0,filename=None):
        df = chunk_wals(self.cols,True,False,self.allow_empty)
        if filename is None:
            filename = "-".join(self.cols + [str(self.numrows)])+'.csv'
        if binary:
            active = pd.get_dummies(df.ix[:,10:])
            supp = df.ix[:,:10]
            df = pd.concat([supp,active],axis=1)
        df.to_csv(os.path.join(ColGroup.csv_dir,filename),ignore_index=True)
        return df

    def __str__(self) :
        ret =  """{0:d} long group covering {1:d} languages
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
            "\n\r".join(self.colnames)
        )
        if isinstance(self.silhouette_score,float):
            top2 = self.families.most_common(2)
            ret += """
family1: {0:d} ({1:s}), 
family2: {2:d} ({3:s}), 
separation: {4:.2f}""".format(
            top2[0][1],
            top2[0][0],
            top2[1][1],
            top2[1][0],
            self.silhouette_score
        )
        return ret

if __name__ == '__main__':
    opts,args = parser.parse_args()
    minrows  = int(args[0])
    loc = Locator(minrows,**opts.__dict__)
    loc.main()
    
