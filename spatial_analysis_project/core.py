import numpy as np
import pandas as pd
import os

class Core:
    def __init__(self, csv_file=None):
        if csv_file is not None:
            self.load_from_csv(csv_file)

    # Load single cell map data from csv file
    def load_from_csv(self, csv_file):
        self.df = pd.read_csv(csv_file)

        # Name of core and cell coordinates
        self.name = os.path.basename(csv_file).split(".")[0]
        self.cell_coordinates = self.df.loc[:, ["Centroid X px", "Centroid Y px"]].values

        self.area = 0.2749**2 * np.pi * np.prod(np.max(self.cell_coordinates, axis=0) - np.min(self.cell_coordinates, axis=0))/4
    
        # Cell types and their integer encoding
        self.cell_types = self.df.loc[:, "Class Type"].values.tolist()
        self.cell_types_set = sorted(list(set(self.cell_types)))
        self.cell_type_dic = dict(zip(self.cell_types_set, np.arange(len(self.cell_types_set))))
        self.cell_types = np.array([self.cell_type_dic[cell] for cell in self.cell_types])
        self.cell_type_number = dict(zip(self.cell_types_set, [np.sum(self.cell_types == i) for i in np.arange(len(self.cell_types_set))]))
        self.cell_number = self.cell_types.shape[0]

        # dictionary for all biomarkers that are implemented
        self.biomarkers = {}

    # Load patient information from csv file
    def load_patient_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)

        # Check if core is contained in csv file
        if df.loc[df["Core ID"] == self.name].empty:
            raise ValueError("Patient information not found for core " + self.name) 
        
        # Set patient information
        self.patient_id = df.loc[df["Core ID"] == self.name]["Patient ID"].values[0]
        self.patient_months = df.loc[df["Core ID"] == self.name]["OS(Months)"].values[0]
        self.patient_status = df.loc[df["Core ID"] == self.name]["Status"].values[0]

    ### Patient information setters ###
    def set_patient_id(self, patient_id):
        self.patient_id = patient_id

    def set_patient_months(self, months):
        self.patient_months = months

    def set_patient_status(self, status):
        self.patient_status = bool(status)

    ##################
    ### Biomarkers ###
    ##################
    def cell_type_fraction(self):
        returns = {}
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_fraction"] = self.cell_type_number[cell_type] / self.cell_number
            returns[cell_type + "_fraction"] = self.biomarkers[cell_type + "_fraction"]
        return returns

    def cell_type_density(self):
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_density_mu^2"] = self.cell_type_number[cell_type] / self.area
        
if __name__ == "__main__":
    a = Core()
    a.load_from_csv('./M06/Predicted Texts/A-1.csv')
    a.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
    a.cell_type_fraction()
    a.cell_type_density()
    print(a.biomarkers)