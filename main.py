import yaml,logging,numpy,pandas,time
from optparse import OptionParser
from feature import Feature,languages
feature_specs = yaml.load(open('features.yml'))
# we might as well create Feature object
# out of all of them
allfeatures = [Feature(name,data) for name,data in feature_specs.items()]

# check what to print #
parser = OptionParser()
parser.add_option("-q", "--quiet",action="store_true",dest="supress_warnings",default=False,help="set logging level to ERROR")
options,args = parser.parse_args()
if options.supress_warnings:
    logging.basicConfig(level=logging.ERROR)

def generate_matrix(saveas,**kwargs):
    """
    :filename: no extension, name of file in which to save the data
    :include: a list of feature names (dict keys features.yml)
    :exclude: same
    if both absent, all the features listed in the features.yml
    will be used
    """
    include = kwargs.get('include')
    exclude = kwargs.get('exclude')
    filename = saveas or time.asctime().replace(' ','_')
    
    target_features = allfeatures
    if include is not None:
        target_features = [f for f in target_features if f.name in include]
    if exclude is not None:
        target_features = [f for f in target_features if not f.name in exclude]
    
    # now,try to find languages that have values for all these features
    target_languages = languages.index
    for feat in target_features:
        target_languages = numpy.intersect1d(feat.get_non_empty(),target_languages)
    if len(target_languages) > 0:
        print("found {} language ids:\n".format(len(target_languages)))
        print(target_languages)
        base_matrix = numpy.zeros((0,len(target_languages)))
        for feat in target_features:
            base_matrix = numpy.vstack((base_matrix,feat.get_languages(target_languages)))
        df = pandas.DataFrame(base_matrix.T,columns = [feat.name for feat in target_features],index = [languages.iloc[l]['wals_code'] for l in target_languages])
        df.to_csv(filename+'.csv')

