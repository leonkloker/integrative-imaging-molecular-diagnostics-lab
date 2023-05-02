from core import Core
from dataset import Dataset

ds = Dataset('./M06/Predicted Texts/')
ds.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
ds.calculate_biomarker()
ds.log_rank_test()
print(ds.log_rank_p)
ds.kaplan_meier("Lymphocyte_average_area_px^2")
