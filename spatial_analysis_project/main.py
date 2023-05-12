from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load()
ds.univariate_cox_model()
pprint(ds.cox_p)
ds.save()
