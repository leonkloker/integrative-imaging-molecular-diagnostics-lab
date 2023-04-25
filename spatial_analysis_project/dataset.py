import kaplanmeier
import matplotlib.pyplot as plt
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

        self.patient_months = np.array(self.patient_months)
        self.patient_status = np.array(self.patient_status)

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
    def kaplan_meier(self, *args):
        for arg in args:
            try:
                biomarker_values = np.array(self.biomarkers[arg])

            except KeyError:
                print("Biomarker " + arg + " not defined or not calculated yet!")
                continue
            
            biomarker_mean = np.mean(biomarker_values)
            group0 = biomarker_values > biomarker_mean
            results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            kaplanmeier.plot(results, savepath="./"+arg+"_kaplan_meier.png", title="Kaplan-Meier curve for {} \n Log-rank p-value = {:.4f}".format(arg, results['logrank_P']))
