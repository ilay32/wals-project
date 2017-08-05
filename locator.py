import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as km
from sklearn.metrics import silhouette_score as silsc
from scipy.spatial.distance import hamming
import copy,fbpca,yaml,hashlib,pickle,nltk,sys,nltk,optparse,os,prince
from progress import progress
from matplotlib import pyplot as plt
from spectral import kmeans

# for cli usage #
#---------------#
parser = optparse.OptionParser()

parser.add_option(
    "-f",
    "--feature-mode",
    dest="fmode",
    metavar="FEATURE_MODE",
    help="shuffle the WALS feature values so that distribution is maintained but the features are rendered bogus"
)

parser.add_option(
    "-s",
    "--silhouette-score-cutoff",
    dest="sil_cutoff",
    metavar="SIL_CUTOFF",
    help="flag (and save) only results with silhoette score larger than SIL_CUTOFF in addition to the CUTOFF restriction."
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
areas = yaml.load(open('wals-areas.yml'))
phonsubs = yaml.load(open('phonology-subareas.yml'))
polinsk = yaml.load(open('polinsk.yml'))

bogwals = pd.DataFrame()

binarized = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
binarized = binarized.replace(to_replace="",value=0)


featfullnames = wals.columns[10:]
feature2area = dict()
code2feature = dict()
for area,codes in areas.items():
    for code in codes:
        feature2area[code] = area
for name in featfullnames:
    code = name.split(" ")[0]
    code2feature[code] = name
    wals = wals.rename(columns={name : code})
    binarized = binarized.rename(columns={name : code})

# this patches up the Zapotec (Zoogocho) row
# on basis of other zapotec languages.
zapvals =  [
    ('84A', '1 VOX'), #by the Mitla dialect
    ('85A', '2 Prepositions'), #by the Isthmus,Yatzatchi and Mitla dialects
    ('86A', '2 Noun-Genitive'), #by the Isthmus,Yatzatchi and Mitla dialects
    ('91A', '1 Degree word-Adjective'),#by the Isthmus dialect
]
for feat,val in zapvals:
    wals.loc[wals['Name'] == 'Zapotec (Zoogocho)',feat] = val



# helpers #
#---------#
def phonsub(f):
    if f == '3A':
        return 'consontants and vowels'
    for sub,feats in phonsubs.items():
        if f in feats:
            return sub

def like(str1,str2,strict=True):
    for sub1 in str1.split(" "):
        for sub2 in str2.split(" "):
            if strict:
                cond = sub1.lower() == sub2.lower()
            else:
                cond = (sub1 in sub2) or (sub2 in sub1)
            if cond:
                return True
    return False

def rebinarize():
    b = wals.ix[:,10:].replace(to_replace=".+",regex=True,value=1)
    b = b.replace(to_replace="",value=0)
    b = b.rename(columns={name : code})
    return b

def verify_areas(l):
    for f in l:
        if f not in areas.keys():
            return False
    return True

def verify_features(l):
    for f in l:
        if f not in code2feature.keys():
            return False
    return True


def subs(l):
    for item in l:
        cop = copy.deepcopy(l)
        cop.remove(item)
        yield cop

def bogify_column(f):
    series = wals[f]
    cats = series.unique()
    langs = series[series != ""]
    ret = pd.Series("",index = series.index)
    used = pd.Index([])
    for i,cat in enumerate(cats):
        if cat == "":
            continue
        num = np.count_nonzero(series == cat)
        inds = np.random.choice(langs.index.difference(used),num,replace=False)
        ret.loc[inds] = "bogus-{:s}-{:d}".format(f,i)
        used = used.union(inds)
    return ret

def bogify_wals():
    ret = wals.ix[:,0:10]
    for c in binarized.columns:
        ret[c] =  bogify_column(c)
    return ret

def chunk_wals(columns,just_actives=True,allow_empty=0):
    global binarized
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
    cols = columns if just_actives else np.concatenate((wals.columns[0:10],columns))
    return wals.loc[indices][cols]

# helper dependant globals #
#-------------------------#
#headclass_indices = islang(sum([l for c,l in polinsk['heads'].items()],[]))
#ratioclass_indices = islang(sum([l for c,l in polinsk['ratios'].items()],[]))

class Locator:
    def __init__(self,minrows,cutoff=2,sil_cutoff=0.5,limit=None,heterogenous=None, \
        include=None,exclude=None,with_clustering=False,allow_empty=0, \
        target_langs=None,target_genuses=None,loi=None,fmode='true'):
        self.minrows = minrows
        self.cutoff = float(cutoff or 2)
        self.sil_cutoff = sil_cutoff
        if with_clustering and with_clustering not in ['genetic','ratio']:
            print('with_clustering (-w) is either "genetic" or "ratio"')
            quit()
        if fmode == 'bogus':
            global wals,bogwals
            wals = bogify_wals()
            bogwals = wals
        self.fmode = fmode
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
        self.target_genuses = self.verify_genuses(target_genuses)
        if loi is not None:
            lois = loi if isinstance(loi,list) else [loi]
            self.loi = wals.loc[wals['Name'].isin(lois)].index
        else:
            self.loi = None
        self.totalfeatures = 0

    def verify_genuses(self,genuses):
        if genuses is None:
            return None
        for g in genuses:
            gcount = np.count_nonzero(wals['genus'] == g)
            if gcount == 0:
                print(g,"is not a WALS genus")
                quit()
            if gcount < self.minrows:
                print("there are only",gcount,"languages of genus",g)
                quit()
        return genuses

    def satisfies(self,cols):
        l = len(cols)
        minrows = self.minrows
        sums = np.sum(binarized[cols],axis=1)
        subtract = self.allow_empty if l > self.allow_empty else 0
        fullrows = sums[sums >= (l - subtract)]
        if self.loi is not None:
            if len(fullrows.index.intersection(self.loi)) == 0:
                return False
        if self.target_langs is not None:
            fullrows = fullrows[fullrows.index.intersection(self.target_langs)]
        if self.target_genuses is not None:
            inds = pd.Index([])
            for genus in self.target_genuses:
                langs = wals[wals['genus'] == genus].index
                inter = fullrows.index.intersection(langs)
                if len(inter) < minrows:
                    return False
                inds = inds.union(inter)
            fullrows = fullrows[inds]

        # if the heterogeneity requirement is on, and the group is too homogenous,
        # check if there's a subset of the covered languages that's larger than minrows and does not
        # include features from the dominant field
        if self.heterogenous is not None and len(fullrows) > minrows:
            fields = nltk.FreqDist([feature2area[c] for c in cols])
            if fields.freq(fields.max()) > self.heterogenous:
                for feature in binarized.columns:
                    if feature2area[feature] != fields.max() and feature not in cols:
                        sums2 = np.sum(binarized[cols + [feature]],axis=1)
                        fullrows2 = len(sums2[sums2 == l + 1])
                        if fullrows2 > minrows:
                            return True
                # the loop completed so this group can't be extended with non-dominant fields
                if len(self.hetreject) > l:
                    self.hetreject[l] += 1
                else:
                    self.hetreject.append(1)
                return False
        return len(fullrows) >= minrows



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

    def main(self,filename=None):
        global binarized
        binarized = rebinarize()
        flagged = list()
        asessed = set()
        qs = dict()
        savefile = os.path.join("feature-sets","colgroups-{:d}".format(self.minrows))
        if self.include is not None:
            inc = self.include if isinstance(self.include,list) else [self.include]
            if verify_areas(inc):
                binarized = binarized[[c for c in binarized.columns if feature2area[c] in inc]]
            elif verify_features(inc):
                binarized = binarized[inc]
            else:
                print("include/exclude must be a WALS area name (lowercase,underscored) or a feature (just the code e.g 1A) or a list of such strings")
                quit()
            savefile += "-inc-{}".format("-".join(inc))

        if self.exclude is not None:
            exc = self.exclude if isinstance(self.exclude,list) else [self.exclude]
            if verify_areas(exc):
                binarized = binarized[[c for c in binarized.columns if feature2area[c] not in exc]]
            elif verify_features(exc):
                binarized = binarized[[c for c in binarized.columns if c not in exc]]
                savefile += "-exc-{}".format("-".join(exc))
        self.totalfeatures = len(binarized.columns)
        allcols = self.locate_all_columns()
        for i in range(2,len(allcols)):
            print("asessing {:d} long groups".format(i+1))
            icols = allcols[i]
            for j,cols in enumerate(icols):
                progress(j,len(icols),min(len(icols),50),"flagged:{:d}".format(len(flagged)))
                groupkey = "".join(sorted(cols))
                if groupkey not in asessed:
                    group = ColGroup(cols,allow_empty=self.allow_empty,fmode=self.fmode)
                    group.determine_spectral_data()
                    qind = group.quality_index
                    k = str(i+1)+'-long'
                    if k in qs:
                        qs[k].append(qind)
                    else:
                        qs[k] = [qind]
                    if  qind > self.cutoff:
                        if self.with_clustering in ['ratio','genetic']:
                            sil = group.gen_separation() if self.with_clustering == 'genetic' else group.ratio_separation()
                        if (self.with_clustering and sil >= self.sil_cutoff) or not self.with_clustering:
                            flagged.append(group)
                    asessed.add(groupkey)
            print()
        if self.heterogenous:
            savefile += "-heterogenous-{:.1f}".format(self.heterogenous)
        if self.limit:
            savefile += "-randlimit-{:.1f}".format(self.limit)
        if filename is not None:
            savefile = os.path.join('feature-sets',filename)
        if len(flagged) > 0 and filename != 'discard':
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
    colors = ['r','g','b','c','k','y']
    def __init__(self,cols,allow_empty=0,fmode='true'):
        self._mode = 'pca'
        self._fmode = fmode
        self.original_fmode = fmode
        self.cols = cols
        self.allow_empty = allow_empty
        self.colnames = [code2feature[c] for c in cols]
        self.numcols = len(cols)
        self.numrows = len(self.get_table())
        self.fields = nltk.FreqDist([feature2area[c] for c in self.cols])
        self._silhouettes = dict()
        self.mca = None
        self.pca = None
        self.families = nltk.FreqDist(self.get_table()['family'])
        consistent = sorted(self.families.most_common(),key=lambda p: p[0])
        consistent.sort(key = lambda p: p[1],reverse=True)
        self.consistent_families = consistent
        self.quality_index = None
        self.dim1 = None
        self.dim2 = None
        self.known_vnratios = 0
        self.phonology = None
        self.current_axis = None
        self.categories = None
        self.bogus_seed = int("".join([c for c in "".join(self.cols) if c.isdigit()])) % (2**32 - 1)

    def set_spectral_core(self,force_new=False):
        if self.mode == 'mca':
            if not isinstance(self.mca,Exception):
                if (self.mca is None) or force_new:
                    try:
                        self.mca = prince.MCA(self.get_table()[self.cols],n_components=-1)
                        self.categories = self.mca.P.columns
                    except Exception as e:
                        self.mca = e
                        print(str(e))
                        return None
        elif not isinstance(self.pca,Exception):
            if (self.pca is None) or force_new:
                try:
                    d = pd.get_dummies(self.get_table()[self.cols])
                    U,s,V = fbpca.pca(d,d.shape[1],raw=False)
                    self.pca = np.square(s),V
                    self.categories = d.columns
                except Exception as e:
                    self.pca = e
                    print(str(e))
                    return None

    def require_pc(fn):
        def wrap(self,*args,**kwargs):
            self.set_spectral_core()
            return fn(self,*args,**kwargs)
        return wrap

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self,mode):
        self._mode = mode
        self.determine_spectral_data()

    @property
    def silhouettes(self):
        if self.mode not in self._silhouettes:
            self._silhouettes[self.mode] = pd.DataFrame(columns=np.arange(len(self.categories)+2))
        return self._silhouettes[self.mode]

    @property
    def fmode(self):
       return self._fmode

    @fmode.setter
    def fmode(self,features_mode):
        self._fmode = features_mode
        if features_mode == 'bogus':
            global bogwals
            if bogwals.empty:
                print('bogifying wals')
                bogwals = bogify_wals()
        self.set_spectral_core(True)

    #legacy, if we gauge importance by bare pca on dummies, let the index be so too
    def mca_asess(self):
        df = self.get_table()[self.cols]
        numvars = self.numcols
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
            return (None,None,None)

    def phonology_subareas(self):
        phon = dict()
        for sub,feats in phonsubs.items():
            phon[sub] = [c for c in self.cols if c in feats]
        self.phonology = phon

    def phonology_spread(self):
        if self.phonology is None:
            self.phonology_subareas()
        submax,subcount = 0,0
        for sub,feats in self.phonology.items():
            if len(feats) > 0:
                subcount += 1
            if len(feats) > submax:
                submax = len(feats)
        return submax/self.numcols,subcount

    @require_pc
    def asess(self):
        if self.mode == 'mca':
            ei = self.mca.explained_inertia
            return ei[0]*self.mca._Base__k, ei[0],ei[1]
        elif self.mode == 'pca':
            s,V = self.pca
            tot = sum(s)
            return  V.shape[0] * s[0] / tot, s[0]/tot, s[1]/tot

    def determine_spectral_data(self):
        asessment = self.asess()
        self.quality_index = asessment[0]
        self.dim1  = asessment[1]
        self.dim2 = asessment[2]

    def fields_spread(self):
        return self.fields.most_common(1)[0][1]/float(self.fields.N())

    def get_table(self):
        table =  chunk_wals(self.cols,False,self.allow_empty)
        if self.fmode == 'bogus':
            table[self.cols] = bogwals[self.cols]
        return table

    def genus_separation(self,target_genuses):
        df = self.get_table()
        labels = df['genus'][df['genus'].isin(target_genuses)]
        self.silhouettes.loc['genus'] = self.compute_silhouettes(labels)
        genuses = nltk.FreqDist(df['genus'])
        #self.target_genuses_counts = [(g,genuses[g]) for g in target_genuses]
        return self.best_silhouette('genus')[0]

    def best_silhouette(self,kind,n_clusts=None):
        k = kind
        if  n_clusts is not None:
            if kind +'-'+str(n_clusts) in self.silhouettes.index:
                k = kind + '-' + str(n_clusts)
            else:
                for i in range(15):
                    if kind + '-' + str(i) in self.silhouettes.index:
                        k = kind + '-' + str(i)
                        break
        if k not in self.silhouettes.index:
            print("must run the",kind,"method first")
            return None
        sils = self.silhouettes.loc[k]
        return sils.max(),sils.argmax() - 1

    def gen_separation(self,n_clusts=2):
        df = self.get_table()
        topfams = [fam[0] for fam in self.consistent_families[:n_clusts]]
        labels = df['family'][df['family'].isin(topfams)]
        self.silhouettes.loc['genetic-'+str(n_clusts)] = self.compute_silhouettes(labels)
        return self.best_silhouette('genetic',n_clusts)[0]

    def plot_silhouettes(self,kind='genetic',clusts=2):
        thresh = 0.5
        sils = self.silhouettes.loc[kind+'-'+str(clusts)].values
        maxdim = len(sils) - 1
        sd = self.significant_dimensions(thresh)
        plt.ylabel('average silhouette coefficient')
        plt.xlabel('dimensions used')
        x = np.arange(-1,len(sils) - 1)

        plt.scatter(x[0],sils[0],color='g',label='without MCA (Eucledian)')
        plt.scatter(x[1],sils[1],color='r',label='without MCA (Hamming)')

        plt.plot(x[2:],sils[2:],label='by MCA projection')
        plt.xticks(np.arange(0,maxdim))
        plt.grid()
        plt.vlines([sd],ymin=sils.min(),ymax=sils.max(),linestyles="dashed",label="{:.1f} of the variance explained".format(thresh))
        plt.legend()
        plt.show()

    def loose_ratio_column(self,columns=['Name','genus','family']):
        df = self.get_table()
        if 'verb-noun-ratio' in df.columns:
            df = df.drop('verb-noun-ratio',1)
        ratios = list()
        count = 0
        for i,r in df.iterrows():
            known = False
            for c,l in polinsk['ratios'].items():
                for lang in l:
                    for column in columns:
                        if like(r[column],lang):
                            ratios.append(c)
                            known = True
                            count += 1
                            break
            if not known:
                ratios.append('unknown')
        df.insert(0,'verb-noun-ratio',pd.Series(ratios,index=df.index))
        self.known_vnratios = count
        return df

    def add_ratio_column(self):
        df = self.get_table()
        if 'verb-noun-ratio' in df.columns:
            df = df.drop('verb-noun-ratio',1)
        ratios = list()
        count = 0
        for i,r in df.iterrows():
            known = False
            for c,l in polinsk['ratios'].items():
                for lang in l:
                    if r['Name'] == lang:
                        ratios.append(c)
                        known = True
                        count += 1
                        break
            if not known:
                ratios.append('unknown')
        df.insert(0,'verb-noun-ratio',pd.Series(ratios,index=df.index))
        self.known_vnratios = count
        return df

    @require_pc
    def compute_silhouettes(self,labels):
        silhouettes = list()
        binact = pd.get_dummies(self.get_table()[self.cols])
        filtered = binact.loc[labels.index]
        for v in np.nditer(filtered.values):
            if np.isinf(v) :
                print('inf',v)
            if np.isnan(v):
                print('nan',v)

        silhouettes.append(silsc(filtered,labels))
        silhouettes.append(silsc(filtered,labels,hamming))
        for numdims in range(1,len(self.categories)+1):
            filtered = self.projections(labels.index)[np.arange(numdims)]
            silhouettes.append(silsc(filtered,labels))
        return silhouettes

    @require_pc
    def projections(self,indices):
        if self.mode == 'mca':
            return self.mca.row_principal_coordinates.loc[indices]
        else:
            V = self.pca[1]
            actdum = pd.get_dummies(self.get_table()[self.cols])
            return actdum.loc[indices].dot(V.T)

    def bogus_labels(self,n_clusts):
        fams = [f for f,c in self.consistent_families[:n_clusts]]
        df = self.get_table()
        indpool = df.loc[df['family'].isin(fams)].index
        labels = pd.Series(np.nan,index=indpool)
        np.random.seed(self.bogus_seed)
        used = pd.Index([])
        for i,f in enumerate(fams):
            ind = np.random.choice(indpool.difference(used),self.families[f],replace=False)
            labels.loc[ind] = "bogus-{:d}".format(int(i+1))
            used = used.union(ind)
        return labels

    def bogus_separation(self,n_clusts=2):
        self.silhouettes.loc['bogus-'+str(n_clusts)] = self.compute_silhouettes(self.bogus_labels(n_clusts))
        return self.best_silhouette('bogus',n_clusts)[0]

    def ratio_separation(self):
        df = self.add_ratio_column()
        labels = df.loc[df['verb-noun-ratio'] != 'unknown']['verb-noun-ratio']
        self.silhouettes.loc['ratio'] = self.compute_silhouettes(labels)
        return self.best_silhouette('ratio')[0]

    def add_genetic_data(self,n_clusts=2,raw=False):
        self.gen_separation(n_clusts)
        df = self.get_table()
        dims = self.best_silhouette('genetic',n_clusts)[0]
        topfams = [fam[0] for fam in self.consistent_families[:n_clusts]]
        labels = df['family'][df['family'].isin(topfams)]
        if raw:
            dims = 'raw'
            starts = list()
            active = pd.get_dummies(df[self.cols])
            for f in topfams:
                choice = labels[labels == f].sample(1).index
                starts.append(active.loc[choice].values)
            filtered = np.array([[pd.get_dummies(active).loc[i].values] for i in labels.index])
            print(filtered)
            pred,means = kmeans(filtered,n_clusts,start_clusters=np.array(starts),distance='L1')
            pred = pred.flatten()
            print(pred)
        else:
            filtered = self.projections(labels.index)[np.arange(dims)]
            pred = km(n_clusters=n_clusts,n_init=15).fit_predict(filtered)
        addcolumn = list()
        prediter = iter(pred)
        for i in df.index:
            addcolumn.append(next(prediter) if i in labels.index else 'other')
        df.insert(0,"kmpredict-{}-{}".format(n_clusts,dims),pd.Series(addcolumn,index=df.index))
        return df

    @require_pc
    def significant_dimensions(self,thresh=0.6):
        if self.mode == 'mca':
            for i,cm in enumerate(self.mca.cumulative_explained_inertia,1):
                if cm >= thresh:
                    return i
        else:
            s = self.pca[0]
            tot  = sum(s)
            i = 0
            cm = 0
            while i < len(s):
                cm += s[i]/tot
                if cm >= thresh:
                    return i + 1
                i += 1


    def plot_multifam(self,mincount='auto'):
        self.current_axis = None
        mc = self.families.most_common()
        if mincount == 'auto':
            mincount = mc[4][1]
        fams = [f for f,c in mc if c >= mincount]
        if len(fams) < 2:
            print("not enough families with",mincount,"languages")
            return
        fampairs = [(fams[i],fams[j]) for i in range(len(fams) - 1) for j in range(i+1,len(fams))]
        numfigs = len(fampairs)
        if numfigs > 3:
            nc = 3
            nr = int(np.ceil(numfigs/3))
        else:
            nc = numfigs
            nr = 1
        fig, axes = plt.subplots(nrows=nr, ncols=nc,figsize=(12,12))
        fax = axes.flatten()
        for i,pair in enumerate(fampairs):
            axis = fax[i]
            self.current_axis = axis
            sil = self.plot_families(fams=pair,multi=True)
            axis.set_title("{}/{} silhouette: {:.2f}".format(*pair,sil),fontsize=10)

        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.9])
        plt.suptitle("Family Pairs with More than {:d} Languages ({:s})\n{:s}".format(mincount,self.mode.upper(),self.comps_line()))
        if i < len(fax):
            for a in fax[i+1:]:
                a.set_visible(False)
        plt.show()

    def pcplot(self,points,labels=None,annotate=True):
        multi = False
        points = -1 * points
        ofsmall = 0.03
        ofbig = 0.1
        dat = self.get_table()
        colors = ColGroup.colors

        # figure out where to plot
        if self.current_axis is None:
            fig,ax = plt.subplots(figsize=(15,15))
            self.current_axis = ax
        else:
            multi = True
            ax = self.current_axis

        ax.set_xlim([points.values[:,0].min() - 1,points.values[:,0].max() + 1])
        ax.set_ylim([points.values[:,1].min() - 1,points.values[:,1].max() + 1])
        self.plot_grid()

        # usually the case
        if labels is not None:
            ulabels = list(set(labels))
            if len(ulabels) > len(colors):
                print("please add colors")
                return
            handles = list()
            for i,label in enumerate(ulabels):
                lpoints = points[labels == label]
                x  = lpoints.values[:,0]
                y = lpoints.values[:,1]
                if not multi:
                    handles.append(ax.scatter(x, y, s=50, marker='o',c=colors[i]))
                else:
                    ax.scatter(x,y,s=50,marker='o',c=colors[i])
                if annotate:
                    positions = [
                        (ofsmall,ofsmall),
                        (ofsmall, -0.85 * ofbig),
                        (-1 * ofbig, ofsmall),
                        (-1 * ofbig, -0.85 * ofbig)
                    ]
                    already = nltk.FreqDist()
                    for j,r in lpoints.iterrows():
                        if (r[0],r[1]) in already:
                            pos = positions[already[(r[0],r[1])]%len(positions)]
                        else:
                            pos = positions[0]
                        ofx = r[0]+pos[0]
                        ofy = r[1]+pos[1]
                        ax.text(ofx,ofy,dat.loc[j,'wals_code'],color=colors[i])
                        already.update({(r[0],r[1]) : 1})

            if not multi:
                ax.legend(handles,ulabels)
        else:
            x = points.values[:,0]
            y = points.values[:,1]
            ax.scatter(x,y,s=50,marker='o',c=colors[0])

        # chart grid and main axes
        #ax.grid(True,which='both')
        #ax.spines['bottom'].set_position('zero')
        #ax.spines['right'].set_color('none')
        #ax.spines['top'].set_color('none')
        #ax.spines['left'].set_position('zero')

    def comps_line(self):
        self.determine_spectral_data()
        return "comp1: {0:.2f}%  comp2: {1:.2f}%".format(100*self.dim1,100*self.dim2)

    @require_pc
    def plot_bogus(self,n_clusts=2):
        suptit = "Coloring Languages by Bogus Properties ({})".format(self.mode)
        labels = self.bogus_labels(n_clusts)
        points = self.projections(labels.index)
        sil = self.best_silhouette('bogus',n_clusts)[0]
        self.pcplot(points,labels)
        plt.suptitle("{}\n{}  silhouette: {:.2f}".format(suptit,self.comps_line(),sil))
        plt.show()
    
    @require_pc
    def plot_vars(self):
        dat = self.get_table()
        cvals = self.get_feature_values()
        fig,ax = plt.subplots(figsize=(15,15))
        self.current_axis = ax
        centers = list()
        for c,vals in cvals.items():
            for v in vals:
                members =  dat[dat[c] == v]
                members = -1 * members
                center = self.projections(members.index).mean()[0:2].values
                label = "{:s} {:d}%".format(v,int(100*len(members)/len(dat)))
                centers.append((center,label)) 
        centers.sort(key=lambda x: x[0][0])
        ax.set_xlim(centers[0][0][0] - 1,centers[-1][0][0] + 1)
        centers.sort(key=lambda x: x[0][1])
        ax.set_ylim([centers[0][0][1] - 1,centers[-1][0][1] + 1])
        self.plot_grid()
        for center,label in centers:
            x,y  = center
            ax.scatter(x,y, s=50, marker='s',c='black')
            ax.text(x + 0.02,y - 0.02,label,color='black')
        ax.tick_params(
            axis='both',
            which='both',
            labelbottom='off',
            labelleft='off'
        )
        plt.show()
    
    def plot_grid(self):
        ax = self.current_axis
        # chart grid and main axes
        ax.grid(True,which='both')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')



    @require_pc
    def plot_families(self,fams=None,multi=False):
        if not multi:
            self.current_axis = None
        dat = self.get_table()
        if fams is None:
            fams = [f for f,c in self.consistent_families[:2]]
        labels = dat.loc[dat['family'].isin(fams)]['family']
        suptit = "{} ({})".format(" ".join(fams),self.mode.upper())
        points = self.projections(dat['family'].isin(fams))
        self.pcplot(points,labels)
        sil = silsc(points.values[:,:2],labels)
        if not multi:
            plt.suptitle("{}\n{}  silhouette: {:.2f}".format(suptit,self.comps_line(),sil))
            plt.show()
        else:
            return sil

    @require_pc
    def loadings(self):
        if self.mode == 'pca':
            V = self.pca[1]
            #comps = ['comp.'+str(i) for i in np.arange(V.shape[0])]
            return  pd.DataFrame(V.T,index=self.categories) #,columns=comps)
        elif self.mode == 'mca':
            return self.mca.column_component_contributions*100

    def weights(self,comps=5):
        """
        :comps: number of PCs for which to compute the features' weights
        """
        loadings = self.loadings()
        if loadings is None:
            return None
        ret = dict()
        for c in self.cols:
            entries = loadings.loc[loadings.index.str.startswith(c)]
            ret[c] = np.square(entries[np.arange(comps)]).sum(axis=0)
        return ret

    def to_csv(self,binary=False,allow_empty=0,filename=None):
        df = self.get_table()
        if filename is None:
            filename = "-".join(self.cols + [str(self.numrows)])+'.csv'
        if binary:
            active = pd.get_dummies(df.ix[:,10:])
            supp = df.ix[:,:10]
            df = pd.concat([supp,active],axis=1)
        df.to_csv(os.path.join(ColGroup.csv_dir,filename),ignore_index=True)
        return df
    
    def get_feature_values(self):
        dat = self.get_table()[self.cols]
        vals = dict()
        for c in self.cols:
            vals[c] = list(dat[c].unique())
        return vals

    def __str__(self) :
        top2 = self.consistent_families[:2]
        ret =  """{0:d} long group covering {1:d} languages
in mode {2:s}:
quality index: {3:.2f}
PC1: {4:.0%}
PC2: {5:.0%}
fields: {6:s}
features:
{7:s}
family1: {8:d} ({9:s})
family2: {10:d} ({11:s})
""".format(
            self.numcols,
            self.numrows,
            self.mode.upper(),
            self.quality_index,
            self.dim1,
            self.dim2,
            self.fields.pformat().replace("FreqDist({","").strip("})"),
            "\n\r".join(self.colnames),
            top2[0][1],
            top2[0][0],
            top2[1][1],
            top2[1][0]
        )
        if  'genetic' in self.silhouettes.index:
            ret += """
genetic separation: {0:.2f} ({1:d} PCs)""".format(*self.best_silhouette('genetic',2))

        if 'ratio' in self.silhouettes.index:
            ret += """
known verb noun ratios: {0:d}
ratio separation: {1:.2f} ({2:d} PCs)""".format(
            self.known_vnratios,
            self.best_silhouette('ratio')[0],
            self.best_silhouette('ratio')[1]
        )
        if 'bogus' in self.silhouettes.index:
            ret += """
bogus separation: {0:.2f} ({1:d} PCs)""".format(*self.best_silhouette('bogus',2))
        return ret

if __name__ == '__main__':
    opts,args = parser.parse_args()
    minrows  = int(args[0])
    loc = Locator(minrows,**opts.__dict__)
    loc.main(filename='discard')

#if __name__ == '__main__':
#    opts = {
#        'include' : areas['phonology'],
#        'cutoff' : 5
#    }
#    loc = Locator(200,**opts)
#    loc.main('discard')
#
