import yaml,logging,numpy
from optparse import OptionParser
from feature import Feature
features = yaml.load(open('features.yml'))
#hardcoding it for now:
nlangs = 2679

# check what to print #
parser = OptionParser()
parser.add_option("-q", "--quiet",action="store_true",dest="supress_warnings",default=False,help="set logging level to ERROR")
options,args = parser.parse_args()
if options.supress_warnings:
    logging.basicConfig(level=logging.ERROR)

# do the main #
base_matrix = numpy.zeros((0,nlangs))
for name,data in features.items():
    feat =  Feature(name,data)
    base_matrix = numpy.vstack((base_matrix,feat.get_languages()))

print(numpy.cov(base_matrix))


