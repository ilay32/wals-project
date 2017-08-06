from locator import *
def stats(dim,groups):
    """
    dim: int specifying the PC to inspect, 0 is first
    """
    ret = dict()
    for g in groups:
        w = g.weights()
        for c in g.cols:
            if c in ret:
                ret[c].append(w[c][dim])
            else:
                ret[c] = [w[c][dim]]
    return ret

def agstats(groups,dims=[0,1]):
    ret = dict()
    df = pd.DataFrame(index = pd.MultiIndex.from_product([['component '+str(d+1) for d in dims], \
    code2feature.keys()],names=['PCs','features']),columns=['mean_loading','std','participation','subarea'])
    for d in dims:
        comp = 'component '+str(d+1)
        stat = stats(d,groups)
        for f,l in stat.items():
            a = np.array(l)
            df.loc[comp,f]['mean_loading'] = a.mean()
            df.loc[comp,f]['std'] = a.std()
            df.loc[comp,f]['participation'] = len(a)
            df.loc[comp,f]['subarea'] = phonsub(f)
    return df
