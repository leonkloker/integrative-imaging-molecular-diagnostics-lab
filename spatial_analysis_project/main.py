from core import Core
from dataset import Dataset
from pprint import pprint

ds = Dataset('./M06/Predicted Texts/')
ds.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
ds.calculate_biomarker("smallest_distance_to_other_types")
ds.calculate_biomarker("amount_of_cells_window")
ds.log_rank_test()
pprint(ds.log_rank_p)