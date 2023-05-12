from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
ds.load()
ds.log_rank_test()
ds.save()
pprint(ds.log_rank_p)
pprint(ds.biomarker_best_cutoff)
