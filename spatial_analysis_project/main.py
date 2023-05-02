from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
ds.calculate_biomarker("average_cell_area")
ds.log_rank_test()
pprint(ds.log_rank_p)
ds.kaplan_meier("Lymphocyte_average_area_px^2")