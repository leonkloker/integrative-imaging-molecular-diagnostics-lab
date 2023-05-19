from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load()
ds.calculate_biomarker("k_function")
ds.save()
ds.calculate_biomarker_mean("k_function")
ds.cox_p("k_function")
ds.log_rank_test("k_function")
ds.save()