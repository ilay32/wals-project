import pandas as pd
import numpy as np
import rpy2.robjects as RobJ
from sklearn.cluster import KMeans as km
from sklearn.metrics import silhouette_score as silsc
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OE
from sklearn.ensemble import RandomForestClassifier as RFC
from scipy.spatial.distance import hamming,euclidean
from scipy.linalg import norm
from scipy.stats import entropy
from progress import progress
from matplotlib import pyplot as plt
from spectral import kmeans
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import copy,fbpca,yaml,hashlib,pickle,nltk,sys,nltk,optparse,os,prince,rpy2


pandas2ri.activate()
importr('randomForest')

# for cli usage #
#---------------#
parser = optparse.OptionParser()

parser.add_option(
    "-t",
    "--asessment_type",
    dest="asess",
    metavar="ASESS",
    help="set asessment criterion. default: 'spectral', other options: None,'families_spread' (will use the -p option if set and None if not)",
)

parser.add_option(
    "-m",
    "--min-freq-fam",
    dest="minfreq_fam",
    metavar="MINFREQF",
    help="set the minimal family count for the most frequent family"
)

parser.add_option(
    "-p",
    "--topfamilies-proportions",
    dest="topfams_prop",
    metavar="(TOP,PROP)",
    help="given the pair (n,max_p), the counts of n most frequent families will be not more than max_p smaller than the top one"
)


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
    "--allow-empty",
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
    "-d",
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


def polinski_headness(walsrow):
    for typ,langs in polinsk['heads'].items():
        for l in langs:
            for c in ['family','genus','Name']:
                if like(walsrow[c],l):
                    return typ
    return 'unknown'

def polinski_ratios(lang):
    for typ,langs in polinsk['ratios'].items():
        for l in langs:
            if like(lang,l):
                return typ
    return 'unknown'


def unique_pairs(l,withsyms=False):
    return [(l[i],l[j]) for i in range(len(l) - 1) for j in range(i+int(not withsyms),len(l))]

def dimrange(d):
    if d > 0:
        return np.arange(d)
    return 0

def phonsub(f):
    if f == '3A':
        return 'consontants and vowels'
    for sub,feats in phonsubs.items():
        if f in feats:
            return sub

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

# helper dependent globals#
headnesses = pd.Series('headnesses',index=wals.index)
vnratios = pd.Series('vnratios',index=wals.index)
hlangs = list()
rlangs = list()
for i,r in wals.iterrows():
    hed = polinski_headness(r)
    rat = polinski_ratios(r['Name'])
    headnesses.loc[i] = hed
    vnratios.loc[i] =  rat  
    if hed != 'unknown':
        hlangs.append(r['Name'])
    if rat != 'unknown':
        rlangs.append(r['Name'])


class Locator:
    def __init__(self,minrows,cutoff=2,sil_cutoff=0.5,limit=None,heterogenous=None, \
        include=None,exclude=None,with_clustering=False,allow_empty=0, \
        target_langs=None,target_genuses=None,loi=None,loi_count=1,fmode='true', \
        topfams_prop=None, minfreq_fam=None,asess='spectral'):
        self.minrows = max(minrows,(minfreq_fam or 0))
        if (topfams_prop is not None) and (minfreq_fam is not None):
            n,p = topfams_prop
            self.minrows = max(int(minfreq_fam + (n-1)*(1-p)*minfreq_fam),minrows)
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
        self.topfams_prop = topfams_prop
        self.minfreq_fam=minfreq_fam
        self.asess = 'families_spread' if (asess=='families_spread' and self.topfams_prop is not None) else None
        if loi is not None:
            lois = loi if isinstance(loi,list) else [loi]
            self.loi = wals.loc[wals['Name'].isin(lois)].index
            self.loi_count = loi_count
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
        if (self.minfreq_fam is not None) and len(fullrows) >= minrows:
            famfreqs = nltk.FreqDist(wals.iloc[fullrows.index]['family']).most_common()
            if famfreqs[0][1] < self.minfreq_fam:
                return False
            if self.topfams_prop is not None:
                n,p = self.topfams_prop
                if len(famfreqs) < n:
                    return False
                thresh = self.minfreq_fam*(1 - p)
                for f,c in famfreqs[1:n]:
                    if c < thresh:
                        return False
        if self.loi is not None:
            if len(fullrows.index.intersection(self.loi)) < self.loi_count:
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
        #print("entering",minrows,numcols)
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
        if self.asess is None:
            flagged = [ColGroup(g) for g in sum(allcols,[])]
            self.flagged = flagged
            return
        else:
            for i in range(2,len(allcols)):
                print("asessing {:d} long groups. criterion:{:s}".format(i+1,self.asess))
                icols = allcols[i]
                for j,cols in enumerate(icols):
                    progress(j,len(icols),min(len(icols),50),"flagged:{:d}".format(len(flagged)))
                    groupkey = "".join(sorted(cols))
                    if groupkey not in asessed:
                        group = ColGroup(cols,allow_empty=self.allow_empty,fmode=self.fmode)
                        if self.heterogenous is not None and group.fields_spread() > self.heterogenous:
                            continue
                        if self.asess == 'families_spread':
                            if g.families_spread(*self.topfams_prop):
                                flagged.append(g)
                        else:
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
    #colors = ['r','g','b','c','k','y']
    colors = [
        'darkviolet',
        'seagreen',
        'lightcoral',
        'dodgerblue',
        'firebrick',
        'gold',
        'charteuse',
        'orange',
        'orchid',
        'sienna',
        'blue',
        'gray',
        'pink'
    ]
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numer_trnstable = str.maketrans(dict([(c,str(i)) for i,c in enumerate(alph,1)]))
    def __init__(self,cols,allow_empty=0,fmode='true',fams_min=None,silhouette_dims=None):
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
        self.minimal_family_strength = consistent[min(4,len([p for p in consistent if p[1] > 1]) - 1)][1]
        self.quality_index = None
        self.dim1 = None
        self.dim2 = None
        self.known_vnratios = 0
        self.phonology = None
        self.current_axis = None
        self.categories = None
        self.bogus_seed = int("".join([c for c in "".join(self.cols) if c.isdigit()])) % (2**32 - 1)
        self.raw_silhouettes = None
        self.paired_silhouettes = None
        self.polinsk_pairwise = {'headness':None,'ratio':None}
        self.silhouette_dims = silhouette_dims
        self.families_rf = None
        self.rforest = None
        self.family_rfs_data = dict()

    def numeric_key(self):
        return int("".join(self.cols).translate(ColGroup.numer_trnstable)) % 2**32

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
    @require_pc
    def silhouettes(self):
        if self.silhouette_dims == 'full':
            dims = len(self.categories) + 2
        else:
            dims = self.significant_dimensions() + 3
        if self.mode not in self._silhouettes:
            self._silhouettes[self.mode] = pd.DataFrame(columns=np.arange(dims))
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

    def families_spread(self,n_fams,max_diff):
        thresh = self.consistent_families[0][1]*(1 - max_diff)
        for f,c in self.consistent_families[1:n_fams]:
            if c < thresh:
                return False
        return True

    def get_table(self):
        table =  chunk_wals(self.cols,False,self.allow_empty)
        if self.fmode == 'bogus':
            table[self.cols] = bogwals[self.cols]
        return table

    def genus_separation(self,target_genuses):
        df = self.get_table()
        labels = df.loc[df['genus'].isin(target_genuses)]['genus']
        self.silhouettes.loc['genus'] = self.compute_silhouettes(labels)
        genuses = nltk.FreqDist(df['genus'])
        #self.target_genuses_counts = [(g,genuses[g]) for g in target_genuses]
        return self.best_silhouette('genus')[0]
    
    def plot_genuses(self,target_genuses):
        df = self.get_table()
        labels = df.loc[df['genus'].isin(target_genuses)]['genus']
        suptit = "{} ({})".format(" ".join(target_genuses),self.mode.upper())
        points = self.projections(df['genus'].isin(target_genuses))
        self.pcplot(points,labels)
        sil = silsc(points.values[:,:2],labels)
        plt.suptitle("{}\n{}  silhouette: {:.2f}".format(suptit,self.comps_line(),sil))
        plt.show()
    
    def polinsk_series(self,prop='headness'):
        if prop not in ['headness','ratio']:
            return
        s = headnesses if prop == 'headness' else vnratios
        select = s[s!='unknown']
        return s[self.get_table().index.intersection(select.index)]
    
    @require_pc
    def polinsk_separation(self,prop='headness',min_clust=5,pairwise=False,classes=None):
        s = self.polinsk_series(prop)
        freqs = nltk.FreqDist(s)
        if classes is None:
            classes = [p[0] for p in freqs.most_common() if p[1] >= min_clust]
            n_clusts = len(classes)
            if pairwise:
                if self.polinsk_pairwise[prop] is None:
                    sep = pd.DataFrame(index=np.arange(int(n_clusts*(n_clusts - 1)/2)),columns=['class1','class2','bestsil','dims'])
                    for i,c1 in enumerate(classes[:-1]):
                        for j,c2 in enumerate(classes[i+1:],1):
                            s = self.polinsk_series(prop)
                            sils = self.compute_silhouettes(s[s.isin([c1,c2])])
                            best = max(sils)
                            dims = max(0,sils.index(best) - 1)
                            sep.loc[i*(n_clusts-1)+j-1] = [c1,c2,best,dims]
                    self.polinsk_pairwise[prop] = sep
                return self.polinsk_pairwise[prop]['bestsil'].mean()
        if prop + '-' +  str(n_clusts) not in self.silhouettes.index:
            if n_clusts < 2:
                print("not enough languages with known",prop)
                return None
            self.silhouettes.loc[prop+'-'+str(n_clusts)] = self.compute_silhouettes(s[s.isin(classes)])
        return self.best_silhouette(prop,n_clusts)[0]

    def plot_polinsk_prop(self,prop='headness',classes=None,min_clust=5):
        self.current_axis = None
        df = self.get_table()
        s = self.polinsk_series(prop)
        if classes is None: 
            freqs = nltk.FreqDist(s)
            classes = [p[0] for p in freqs.most_common() if p[1] >= min_clust]
        suptit = "Separation by {:s} Class ({:s})".format(prop,self.mode.upper())
        labels = s[s.isin(classes)]
        points = self.projections(labels.index)
        self.pcplot(points,labels)
        sil = silsc(points.values[:,:2],labels)
        plt.suptitle("{:s}\n{:s}  silhouette: {:.2f}".format(suptit,self.comps_line(),sil))
        plt.show()
    
    def bogus_separation(self,n_clusts=2):
        self.silhouettes.loc['bogus-'+str(n_clusts)] = self.compute_silhouettes(self.bogus_labels(n_clusts))
        return self.best_silhouette('bogus',n_clusts)[0]

    
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
        sils = self.silhouettes.loc[k][2:]
        return sils.max(),sils.argmax() - 1
    
    def single_family_labels(self,family,data=None,seedplus=0):
        df = self.get_table() if data is None else data
        np.random.seed(self.numeric_key() + seedplus)
        tar = df[df['family'] == family]['family']
        others = df[df['family'] != family]['family'].sample(n=len(tar)).apply(lambda f: 'other')
        return pd.concat([tar,others]).sort_index()
    
    def rforest_command(self,forestargs):
        ret = 'randomForest(family ~ ., data=lapply(rforestdat,as.factor),importance=TRUE'
        if forestargs is None:
            forestargs = {
                'ntree' : 200,
                'mtry' : max(len(self.cols) - 2, 2)
            }
        for k,v in forestargs.items():
            ret += ','+str(k)+'='+str(v)
        return ret+')'

    def rrf_single(self,top=0,fam=None,**forestargs):
        if fam is None:
            fam = self.consistent_families[top][0]
        sp= forestargs.get('seedplus') or 0
        family = self.single_family_labels(fam,data=None,seedplus=sp)
        df = self.get_table()[self.cols].loc[family.index]
        df['family'] = family
        rdf = RobJ.pandas2ri.py2ri(df)
        RobJ.globalenv['rforestdat'] = rdf
        forest  = RobJ.r(self.rforest_command(forestargs))
        self.rforest = forest
        RobJ.globalenv['rforestdat'] = RobJ.NULL

    def family_rfs(self,family,numforests=5):
        dat =  pd.DataFrame(index=['sample '+str(i) for i in range(numforests)],columns=['oob','precision','recall','f-measure','rerror'])
        for i in range(numforests):
            self.rrf_single(fam=family,seedplus=i)
            oob = self.get_rforest_data('err.rate').iloc[-1][0]
            conf = self.get_rforest_data('confusion')
            pred = conf.loc[family][family]
            prec = pred/conf[family].sum()
            rec = pred/conf[[family,'other']].loc[family].sum()
            rerr  = conf.loc[family]['class.error']
            fmeas  =  (2 * prec * rec)/(prec + rec)
            dat.iloc[i] = [oob,prec,rec,fmeas,rerr]
        self.family_rfs_data[family] = dat.astype(float)
    
    def get_rforest_data(self,item,forestargs=None):
        if self.rforest is None:
            self.rrf_single(**forestargs)
        RobJ.globalenv['forest'] = self.rforest
        item = 'forest$'+item
        it = RobJ.r(item)
        rows = RobJ.r('rownames('+item+')')
        cols = RobJ.r('colnames('+item+')')
        ret = pd.DataFrame(it)
        if rows is not rpy2.rinterface.NULL:
            ret.index = rows
        if cols is not rpy2.rinterface.NULL:
            ret.columns = cols
        RobJ.globalenv['forest'] = RobJ.NULL
        return ret
        
        
    def gen_separation(self,n_clusts=2,family=None):
        df = self.get_table()
        if family in df['family'].values:
            #families = df['family'].apply(lambda f: f if f == family else 'other')
            #count = len(families[families==family])
            #labels = pd.concat([families[families==family],families[families=='other'].sample(n=count)])
            labels = self.single_family_labels(family,data=df)
            self.silhouettes.loc[family] = self.compute_silhouettes(labels)
            return self.best_silhouette(family)[0]
        topfams = [fam[0] for fam in self.consistent_families[:n_clusts]]
        labels = df['family'][df['family'].isin(topfams)]
        self.silhouettes.loc['genetic-'+str(n_clusts) ] = self.compute_silhouettes(labels)
        return self.best_silhouette('genetic',n_clusts)[0]


    def plot_silhouettes(self,kind='genetic',clusts=2,withsingles=False):
        thresh = 0.5
        sils = self.silhouettes.loc[kind+'-'+str(clusts)].values
        maxdim = len(sils) - 1
        sd = self.significant_dimensions(thresh)
        fig,ax = plt.subplots(figsize=(14,12))
        ax.set_ylabel('average silhouette coefficient')
        ax.set_xlabel('dimensions used')
        x = np.arange(-1,len(sils) - 1)

        ax.scatter(x[0],sils[0],color='g',label='without MCA (Eucledian)')
        ax.scatter(x[1],sils[1],color='r',label='without MCA (Hamming)')

        ax.plot(x[2:],sils[2:],label='by MCA projection')
        ax.set_xticks(np.arange(0,maxdim))
        plt.grid()
        ax.vlines([sd],ymin=sils.min(),ymax=sils.max(),linestyles="dashed",label="{:.1f} of the variance explained".format(thresh))
        
        if withsingles and len(self.consistent_families) > 1:
            singles = self.raw_silhouettes.filter(items=[(c,c) for c in self.cols],axis=0)
            top2 = [f for f,c in self.consistent_families[:2]]
            for i,r in singles.iterrows():
                top2sil = r[top2[0],top2[1]]
                if not pd.isnull(top2sil):
                    ax.plot((x[0],x[-1]),(top2sil,top2sil),'-.',label='{} alone'.format(i[0]))
        plt.suptitle('Silhouette Scores on Top {:d} Families'.format(clusts))
        plt.legend()
        plt.show()

    #def loose_ratio_column(self,columns=['Name','genus','family']):
    #    df = self.get_table()
    #    if 'verb-noun-ratio' in df.columns:
    #        df = df.drop('verb-noun-ratio',1)
    #    ratios = list()
    #    count = 0
    #    for i,r in df.iterrows():
    #        known = False
    #        for c,l in polinsk['ratios'].items():
    #            for lang in l:
    #                for column in columns:
    #                    if like(r[column],lang):
    #                        ratios.append(c)
    #                        known = True
    #                        count += 1
    #                        break
    #        if not known:
    #            ratios.append('unknown')
    #    df.insert(0,'verb-noun-ratio',pd.Series(ratios,index=df.index))
    #    self.known_vnratios = count
    #    return df

    #def add_ratio_column(self):
    #    df = self.get_table()
    #    if 'verb-noun-ratio' in df.columns:
    #        df = df.drop('verb-noun-ratio',1)
    #    ratios = list()
    #    count = 0
    #    for i,r in df.iterrows():
    #        known = False
    #        for c,l in polinsk['ratios'].items():
    #            for lang in l:
    #                if r['Name'] == lang:
    #                    ratios.append(c)
    #                    known = True
    #                    count += 1
    #                    break
    #        if not known:
    #            ratios.append('unknown')
    #    df.insert(0,'verb-noun-ratio',pd.Series(ratios,index=df.index))
    #    self.known_vnratios = count
    #    return df

    @require_pc
    def compute_silhouettes(self,labels):
        if len(self.consistent_families) < 2:
            return None
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
        #for numdims in range(1,len(self.categories)+1):
        dims = len(self.categories) if self.silhouette_dims == 'full' else self.significant_dimensions() + 1
        for numdims in range(1,dims+1):
            filtered = self.projections(labels.index)[np.arange(numdims)]
            silhouettes.append(silsc(filtered,labels))
        return silhouettes
    
    def fampairs(self,mincount='auto'): 
        if mincount == 'auto':
            mincount = self.minimal_family_strength
        fams = [f for f,c in self.consistent_families if c >= mincount]
        if len(fams) < 2:
            print("not enough families with",mincount,"languages")
            return
        return unique_pairs(fams)

    def compute_raw_silhouettes(self):
        if self.raw_silhouettes is not None:
            return self.raw_silhouettes
        dat = self.get_table()
        ret = pd.DataFrame(index=pd.MultiIndex.from_tuples(unique_pairs(self.cols,True)),columns=pd.MultiIndex.from_tuples(self.fampairs()))
        for c1,c2 in ret.index:
            d = pd.get_dummies(self.get_table()[list(set([c1,c2]))])
            for fam1,fam2 in ret.columns:
                if fam1 == fam2:
                    sil = 1
                else:
                    labels = dat.loc[dat['family'].isin([fam1,fam2])]['family']
                    filtered = d.loc[labels.index]
                    sil = silsc(filtered,labels,hamming)
                ret.loc[c1,c2][fam1,fam2] = sil
        self.raw_silhouettes = ret
        return ret
    
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

    def add_genetic_data(self,n_clusts=2,raw=False):
        self.gen_separation(n_clusts)
        df = self.get_table()
        dims = self.best_silhouette('genetic',n_clusts)[1]
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
            filtered = self.projections(labels.index)[dimrange(dims)]
            pred = km(n_clusters=n_clusts,n_init=15).fit_predict(filtered)
        addcolumn = list()
        prediter = iter(pred)
        for i in df.index:
            addcolumn.append(next(prediter) if i in labels.index else 'other')
        df.insert(0,"kmpredict-{}-{}".format(n_clusts,dims),pd.Series(addcolumn,index=df.index))
        return df
    
    @require_pc
    def predict(self,lang,prop,clusters):
        df = self.get_table()
        sil = self.best_silhouette(prop,len(clusters))
        if sil is None:
            return None 
        labels = df.loc[df[prop].isin(clusters)]
        langid = labels[labels['wals_code'] == lang].index[0]
        lang_natural_index = labels.index.get_loc(langid)
        filtered = self.projections(labels.index)[dimrange(sil[1])]
        try:
            preds =  km(n_clusters=len(clusters),n_init=15).fit_predict(filtered)
            paired = list(zip(labels[prop],preds))
            return paired[lang_natural_index][0] 
        except Exception as e:
            print(sil)
            return 'yosi'


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
        single = isinstance(self,SingleCol)
        if mincount == 'auto':
            mincount = self.minimal_family_strength
        self.current_axis = None
        pairs = self.fampairs()
        numfigs = len(pairs)
        if numfigs > 3:
            nc = 3
            nr = int(np.ceil(numfigs/3))
        else:
            nc = numfigs
            nr = 1
        fig, axes = plt.subplots(nrows=nr, ncols=nc,figsize=(12,12))
        fax = axes.flatten()
        if not single:
            rsils = self.compute_raw_silhouettes()
        sils = list()
        for i,pair in enumerate(pairs):
            axis = fax[i]
            self.current_axis = axis
            sil = self.plot_families(fams=pair,multi=True)
            sils.append(sil)
            title = "{:s}/{:s} silhouette: {:.2f}".format(*pair,sil)
            if not single:
                singlesils = rsils[pair[0],pair[1]].filter(items=[(c,c) for c in self.cols])
                singlesline = " ".join(["{:s}:{:.2f}".format(c[0],s) for c,s in singlesils.items() if s > sil])
                title += "\nbetter by single feature (hamming):\n{:s}".format(singlesline)
            axis.set_title(title,fontsize=10)
        psils = self.pair_sils()
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.9])
        plt.suptitle("Family Pairs with More than {:d} Languages ({:s})\n{:s} Average silhouette score:{:.2f}".format(mincount,self.mode.upper(),self.comps_line(),np.mean(sils)))
        if i < len(fax):
            for a in fax[i+1:]:
                a.set_visible(False)
        #plt.show()

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
    def plot_vars(self,feats=None):
        if feats is None:
            feats = self.cols
        dat = self.get_table()
        cvals = self.get_feature_values()
        fig,ax = plt.subplots(figsize=(15,15))
        self.current_axis = ax
        centers = list()
        for c in feats:
            vals = cvals[c]
            for v in vals:
                members =  dat[dat[c] == v]
                members =  members
                center = -1 * self.projections(members.index).mean()[0:2].values
                label = "{:s}: {:s} {:d}%".format(c,v,int(100*len(members)/len(dat)))
                centers.append((center,label)) 
        centers.sort(key=lambda x: x[0][0])
        ax.set_xlim(centers[0][0][0] - 1,centers[-1][0][0] + 1)
        centers.sort(key=lambda x: x[0][1])
        ax.set_ylim([centers[0][0][1] - 1,centers[-1][0][1] + 1])
        self.plot_grid()
        for center,label in centers:
            x,y  = center
            ax.scatter(x,y, s=50, marker='s',c='black')
            ax.text(x + 0.03,y,label,color='black')
        ax.tick_params(
            axis='both',
            which='both',
            labelbottom='off',
            labelleft='off'
        )
        plt.show()
        self.current_axis = None
    
    def plot_grid(self):
        ax = self.current_axis
        # chart grid and main axes
        ax.grid(True,which='both')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')


    @require_pc
    def pair_sils(self,pairs=None):
        if len(self.consistent_families) < 2:
            return None
        if self.silhouettes.isnull().all().all():
            return None
        if self.paired_silhouettes is not None:
            return self.paired_silhouettes
        if pairs is None:
            pairs = self.fampairs()
        if pairs is None:
            return None
        dat = self.get_table()
        dims = self.best_silhouette('genetic',2)[1]
        ret = pd.Series('gen2sil',index=pd.MultiIndex.from_tuples(pairs))
        for pair in pairs:
            labels = dat.loc[dat['family'].isin(pair)]['family']
            points = self.projections(dat['family'].isin(pair))
            ret.loc[pair[0],pair[1]] = silsc(points.values[:,:dims],labels)
        self.paired_silhouettes = ret
        return ret
            
    @require_pc
    def plot_families(self,fams=None,multi=False):
        labels = None
        if not multi:
            self.current_axis = None
        dat = self.get_table()
        if fams is None:
            fams = [f for f,c in self.consistent_families[:2]]
        elif isinstance(fams,int):
            fams = [f for f,c in self.consistent_families[:fams]]
        elif isinstance(fams,str) and fams in dat['family'].values:
            labels = self.single_family_labels(fams,data=dat)
            points = self.projections(labels.index)
            suptit = "{1:d} {0:s} vs. {1:d} Random Others ({2:s})".format(fams,self.families[fams],self.mode.upper())
        if labels is None:
            labels = dat.loc[dat['family'].isin(fams)]['family']
            points = self.projections(dat['family'].isin(fams))
            suptit = "{} ({})".format(" ".join(fams),self.mode.upper())
        self.pcplot(points,labels)
        sil = silsc(points.values[:,:2],labels)
        if not multi:
            plt.suptitle("{}\n{}  silhouette: {:.2f}".format(suptit,self.comps_line(),sil))
            #plt.show()
        else:
            return sil

    @require_pc
    def loadings(self):
        if self.mode == 'pca':
            V = self.pca[1]
            #comps = ['comp.'+str(i) for i in np.arange(V.shape[0])]
            return  pd.DataFrame(V.T,index=self.categories) #,columns=comps)
        elif self.mode == 'mca':
            return self.mca.column_component_contributions * 100

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
    
    def silhouettes_repr(self): 
        ret = ""
        for silkind in self.silhouettes.index:
            sils =  self.silhouettes.loc[silkind]
            ret += "{0:s}: {1:.2f} ({2:d} PCs)\n\r".format(silkind,sils.max(),np.argmax(sils) - 1)
        if ret != "":
            ret = "silhouettes:\n\r"+ret
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
        if self.quality_index is None:
            self.determine_spectral_data()
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
{12:s}
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
            top2[1][0],
            self.silhouettes_repr()
        )
        if self.family_rfs_data:
            d = self.family_rfs_data
            for fam,dat in d.items():
                ret += "{0:s} vs others: oob: {1:.3f} f-measure: {2:.3f} ({3:d} samples)\n".format(
                    fam,
                    dat['oob'].mean(),
                    dat['f-measure'].mean(),
                    len(dat)
                )
        return ret
    
    #random forest family classification
    def train_rf(self,n_families=None,cv_proportion=0.85):
        if n_families is None:
            n_families = len([p for p in self.consistent_families if p[1] >= self.minimal_family_strength])
        if not n_families or n_families < 2:
            print("please specify number of families to classify, or minimal family strength not larger than",self.conistent_families[1][1])
            return
        #df,keys = self.prepare_rf_table(n_families)
        X,y = self.prepare_rf_table(n_families)
        rfc = RFC(criterion='entropy')
        train = X.sample(int(len(X)*cv_proportion))
        test = X.loc[X.index.difference(train.index)]
        #rfc.fit(train[self.cols],train['family'])
        rfc.fit(train.loc[:,train.columns!='family'],y.loc[train.index])
        acc = rfc.score(test.loc[:,test.columns!='family'],y.loc[test.index])
        self.families_rf = {
            'cv': {
                'accuracy' : acc,
                'train' : cv_proportion,
                'classifier': rfc
            },
            'fullclassifier' : RFC(criterion='entropy',min_impurity_decrease=0.05).fit(X,y)#,
            #'label_keys' : keys
        }
    
    def prepare_rf_table(self,n_families):
        t = self.get_table()
        reduced = t[t['family'].isin([f for f,c in self.consistent_families[:n_families]])][self.cols + ['family']]
        #labelkeys = dict()
        #for c in self.cols:
        #    le = LE().fit(t[c].unique())
        #    labelkeys[c] = le
        #    ret[c] = le.transform(ret[c])
        #return ret,labelkeys
        return pd.get_dummies(reduced[self.cols]),reduced['family']


        

class SingleCol(ColGroup):
    sqrt2 = np.sqrt(2)
    def __init__(self,wals_feature=None,mincount=10,colgroup=None):
        if wals_feature is None and colgroup is None:
            print("must provide a feature somehow")
            return
        if not isinstance(colgroup,ColGroup):
            super(SingleCol,self).__init__([wals_feature])
            self.feat = wals_feature
        else:
            if len(colgroup.cols) > 1:
                print("construct from ColGroup requires a single feature group")
                return
            self.__dict__ = colgroup.__dict__
            self.feat = colgroup.cols[0]
        if self.families.B() < 2:
            self.mincount = self.families.most_common(1)[0][1]
        else:
            self.mincount = min(mincount,self.consistent_families[1][1])
        self.vals = [ v for v in wals[self.feat].unique() if v != ""]

    def fdist(self,family):
        f = self.feat
        rel = wals.loc[(wals['family'] == family) & (wals[f] != "")][f]
        dist = nltk.FreqDist(rel)
        for val in self.vals:
            if val not in dist:
                dist.update({val:0})
        return dist

    def fdists(self):
        ret = nltk.ConditionalFreqDist()
        for fam,count in self.families.most_common():
            if count > self.mincount:
                ret[fam] = self.fdist(fam)
        return ret

    def plot_multifam_bars(self):
        dists = self.fdists()
        df = pd.DataFrame(index=self.vals,columns=dists.conditions())
        for val in df.index:
            for l in df.columns:
                df.loc[val][l] = dists[l][val]
        df.sort_index(axis=0,inplace=True)
        fig,ax = plt.subplots()
        self.current_axis = ax
        ax.set_title("Distribution of {:s}\non Families with More Than {:d} Languages".format(code2feature[self.feat],self.mincount)) 
        ax.set_xlabel("Feature Values")
        ax.set_ylabel("Number of Languages")
        df.plot.bar(ax=ax,figsize=(15,8))
        plt.show()

    def measures(self,fam1,fam2):
        pd = nltk.LidstoneProbDist(self.fdist(fam1),0.5)
        qd = nltk.LidstoneProbDist(self.fdist(fam2),0.5)
        #p = [pd.freq(v) for v in self.vals]
        #q = [qd.freq(v) for v in self.vals]
        p = [pd.prob(v) for v in self.vals]
        q = [qd.prob(v) for v in self.vals]

        return {
            'symKL':min((entropy(p,q) + entropy(q,p))/2,10), 
            'euc' : euclidean(q,p), 
            'hellinger' : norm(np.sqrt(p) - np.sqrt(q)) / SingleCol.sqrt2
        }

    def KLs(self):
        fams = [f for f,c in self.families.most_common() if c >= self.mincount]
        ret = list()
        ind = list()
        for i,f1 in enumerate(fams[0:-1]):
            for j,f2 in enumerate(fams[i+1:]):
                ret.append(self.measures(f1,f2))
                ind.append((f1,f2))
        return pd.DataFrame(ret,index=pd.MultiIndex.from_tuples(tuples=ind,names=['fam1','fam2']))
        

if __name__ == '__main__':
    g = ColGroup(['30A','31A','32A','49A','50A','51A'])
    t,k = g.prepare_rf_table(4)
#    opts,args = parser.parse_args()
#    minrows  = int(args[0])
#    loc = Locator(minrows,**opts.__dict__)
#    loc.main(filename='discard')


