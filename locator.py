import pandas as pd
import numpy as np
import copy
def locate_columns(df,minrows,numcols,cache):
    ret = list()
    if numcols == 1:
        cols = df.columns[df.sum(axis=0) >= minrows]
        ret = [[c] for c in cols]
        cache.append(ret)
    elif len(cache) >= numcols:
        ret =  cache[numcols - 1]
    else:
        ret = list()
        prev = locate_columns(df,minrows,numcols - 1,cache)
        for colgroup in prev:
            for add in df.columns:
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
                        if satisfies(df,newgroup,minrows):
                            ret.append(newgroup)
        if len(ret) > 0:
            cache.insert(numcols,ret)
    return ret

def subs(l):
    for item in l:
        cop = copy.deepcopy(l)
        cop.remove(item)
        yield cop
        
def satisfies(df,colgroup,minrows):
    sums = np.sum(df[colgroup],axis=1)
    return len(sums[sums == len(colgroup)]) >= minrows  
    
def locate_all_columns(df,minrows):
    ret = list()
    for i in reversed(range(1,len(df.columns))):
        if len(locate_columns(df,minrows,i,ret)) == 0:
            break
    return ret
        

if __name__ == '__main__':
    wals = pd.read_csv('language.csv',na_filter=False)
    wals = wals.ix[:,10:]
    wals  = wals.replace(to_replace=".+",regex=True,value=1)
    wals = wals.replace(to_replace='',value=0)
    #test = np.random.binomial(1,0.1,size=(20000,20))
    #test = pd.DataFrame(test)
