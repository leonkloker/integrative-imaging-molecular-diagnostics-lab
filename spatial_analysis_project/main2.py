import dataset
import pickle
import pprint

ds = dataset.Dataset()
ds.load()
pprint.pprint(ds.biomarkers)
