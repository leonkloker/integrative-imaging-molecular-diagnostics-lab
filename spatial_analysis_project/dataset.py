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
        self.log_rank_p = {}

    # Synchronize cell types over all cores
    def sync_cell_types(self):
        cell_types = set()
        for core in self.cores:
            for cell_type in core.cell_types_set:
                cell_types.add(cell_type)
        
        for core in self.cores:
            core.cell_types_set = sorted(list(cell_types))
            for cell_type in core.cell_types_set:
                if not cell_type in core.cell_types_dic.keys():
                    core.cell_types_dic[cell_type] = len(core.cell_types_dic)
                    core.cell_types_number[cell_type] = 0

    # Load all cores from directory
    def load_from_directory(self, directory):
        self.cores = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                core = Core(os.path.join(directory, filename))
                self.cores.append(core)
        self.sync_cell_types()

    # Load patient information from csv file
    def load_patient_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.patient_months = []
        self.patient_status = []

        # Check if core is contained in csv file
        for core in self.cores:
            try:
                core.load_patient_from_csv_(csv_file)
                self.patient_months.append(core.patient_months)
                self.patient_status.append(core.patient_status)

            except ValueError:
                print("Core " + core.name + " not found in patient information csv file")

        self.patient_months = np.array(self.patient_months)
        self.patient_status = np.array(self.patient_status)

    # Calculate expression of given biomarkers for all cores
    def calculate_biomarker(self, *biomarkers):
        if not biomarkers:
            biomarkers = []
            for method in dir(Core):
                if not method.endswith("_"):
                    biomarkers.append(method)

        for biomarker in biomarkers:
            try:
                for i, core in enumerate(self.cores):
                    biomarker_dic = getattr(core, biomarker)()
            
                    for marker in biomarker_dic.keys():
                        if i == 0:
                            self.biomarkers[marker] = [biomarker_dic[marker]]
                        else:
                            self.biomarkers[marker].append(biomarker_dic[marker])

            except AttributeError:
                print("Biomarker " + biomarker + " not defined!")

    # Calculate Kaplan-Meier survival curve for given biomarkers
    def kaplan_meier(self, *biomarkers):
        for biomarker in biomarkers:
            try:
                biomarker_values = np.array(self.biomarkers[biomarker])

            except KeyError:
                print("Biomarker " + biomarker + " not defined or not calculated yet!")
                continue
            
            biomarker_mean = np.mean(biomarker_values)
            group0 = biomarker_values > biomarker_mean
            results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            kaplanmeier.plot(results, savepath="./kaplan_meier/"+biomarker+"_kaplan_meier.png", title="Kaplan-Meier curve for {} \n Log-rank p-value = {:.4f}".format(biomarker, results['logrank_P']))

    # Calculate significance of log-rank test for a given biomarker
    def log_rank_test(self, *biomarkers):
        if not biomarkers:
            biomarkers = self.biomarkers.keys()
        
        for biomarker in biomarkers:
            try:
                biomarker_values = np.array(self.biomarkers[biomarker])

            except KeyError:
                print("Biomarker " + biomarker + " not defined or not calculated yet!")
                continue
                
            biomarker_mean = np.nanmean(biomarker_values)
            group0 = biomarker_values > biomarker_mean

            results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            self.log_rank_p[biomarker] = results['logrank_P']