import pandas,numpy
from extractors import *
languages = pandas.read_csv('language.csv',na_filter=False)

class Feature:
    def __init__(self,name,data):
        self.specs = data
        self.name = name
        self.extractor = eval(data['extractor'])
        self.default = self.specs.get('default') or 0        
    
    def extract(self,cell):
        return self.extractor(cell) if cell != "" else self.default

    def get_languages(self,langids=None):
        langv = list()
        if langids is None:
            langs = languages
        else:
            langs = languages.filter(items=langids,axis=0)
        for cell in langs[self.find_column()].values:
            langv.append(self.extract(cell))
        return numpy.array(langv)
    
    def get_non_empty(self):
        return [i for i in languages.index if languages.iloc[i][self.find_column()] != ""]

    def find_column(self):
        for col in languages:
            if col.startswith(self.specs['wals_id']):
                return col

            
