from core import Core
from dataset import Dataset

a = Dataset('./M06/Predicted Texts/')
a.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
a.biomarker("cell_type_fraction")
a.kaplan_meier("Tumor_fraction")