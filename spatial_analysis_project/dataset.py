import numpy as np
import pandas as pd
import os

from core import Core

class Dataset:
    def __init__(self, directory=None):
        if directory is not None:
            self.load_from_directory(directory)
        self.biomarkers = {}

    # Load all cores from directory
    def load_from_directory(self, directory):
        self.cores = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                core = Core(os.path.join(directory, filename))
                self.cores.append(core)

    # Load patient information from csv file
    def load_patient_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.patient_months = []
        self.patient_status = []

        # Check if core is contained in csv file
        for core in self.cores:
            try:
                core.load_patient_from_csv(csv_file)
                self.patient_months.append(core.patient_months)
                self.patient_status.append(core.patient_status)

            except ValueError:
                print("Core " + core.name + " not found in patient information csv file")

    # Calculate expression of given biomarkers for all cores
    def biomarker(self, *args):
        for arg in args:
            try:
                for i, core in enumerate(self.cores):
                    biomarker_dic = getattr(core, arg)()
                    
                    for marker in biomarker_dic.keys():
                        if i == 0:
                            self.biomarkers[marker] = [biomarker_dic[marker]]
                        else:
                            self.biomarkers[marker].append(biomarker_dic[marker])

            except AttributeError:
                print("Biomarker " + arg + " not defined!")
    
    # Calculate Kaplan-Meier survival curve for given biomarkers
    def kaplan_maier(self, *args):
        for arg in args:
            try:
                biomarker_values = self.biomarkers[arg]

            except KeyError:
                print("Biomarker " + arg + " not defined or not calculated yet!")

if __name__ == "__main__":
    a = Dataset('./M06/Sample Texts/')
    a.load_patient_from_csv('./Patient_Prognostic_Information_v2.csv')
    a.biomarker("cell_type_fraction")
    print(a.biomarkers)
