from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
ds.calculate_biomarker()
ds.log_rank_test()
ds.calculate_biomarker_mean()
pprint(ds.biomarkers_mean)
pprint(ds.log_rank_p)
