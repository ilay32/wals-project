import pandas,numpy
from extractors import *
languages = pandas.read_csv('language.csv',na_filter=False)

class Feature:
    def __init__(self,name,data):
        self.specs = data
        self.name = name
        self.extractor = eval(data['extractor'])
    
    def extract(self,cell):
        return self.extractor(cell) if cell else self.specs['default']

    def get_languages(self):
        langv = list()
        for cell in languages[self.find_column()].values:
            langv.append(self.extract(cell))
        return numpy.array(langv)

    def find_column(self):
        for col in languages:
            if col.startswith(self.specs['wals_id']):
                return col

            
