import kaplanmeier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

from core import Core
from natsort import natsorted
from tqdm import tqdm

class Dataset:
    def __init__(self, directory=None):
        if directory is not None:
            self.load_from_directory(directory)
        self.biomarkers = {}
        self.biomarkers_mean = {}
        self.biomarker_best_cutoff = {}
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
        print("Loading cores from directory " + directory + " ...")
        self.cores = []
        self.cores_name = []
        for filename in natsorted(os.listdir(directory)):
            if filename.endswith(".csv"):
                core = Core(os.path.join(directory, filename))
                self.cores.append(core)
                self.cores_name.append(core.name)
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

    # Calculate mean expression of all calculated biomarkers for all cores
    def calculate_biomarker_mean(self):
        for biomarker in self.biomarkers.keys():
            self.biomarkers_mean[biomarker] = np.nanmean(self.biomarkers[biomarker])

    # Calculate expression of given biomarkers for all cores
    def calculate_biomarker(self, *biomarkers):
        if not biomarkers:
            biomarkers = []
            for method in dir(Core):
                if not method.endswith("_"):
                    biomarkers.append(method)

        for biomarker in biomarkers:
            print("Calculating biomarker " + biomarker)
            try:
                for i in tqdm(range(len(self.cores))):
                    biomarker_dic = getattr(self.cores[i], biomarker)()
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
            
            biomarker_median = np.nanmedian(biomarker_values)
            group0 = biomarker_values > biomarker_median
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
                
            biomarker_median = np.nanmedian(biomarker_values)
            group0 = biomarker_values > biomarker_median
            results_median = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            self.log_rank_p[biomarker] = [results_median['logrank_P']]

            biomarker_mean = np.nanmean(biomarker_values)
            group0 = biomarker_values > biomarker_mean
            results_median = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            self.log_rank_p[biomarker].append(results_median['logrank_P'])

            cutoffs = np.linspace(np.nanmin(biomarker_values), np.nanmax(biomarker_values), 20)
            cutoffs = cutoffs[1:-1]
            log_rank_p = []
            for cutoff in cutoffs:
                group0 = biomarker_values > cutoff
                results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
                log_rank_p.append(results['logrank_P'])

            log_rank_p = np.array(log_rank_p)
            if not np.isnan(log_rank_p).all():
                self.log_rank_p[biomarker].append(np.nanmin(log_rank_p))
                self.biomarker_best_cutoff[biomarker] = cutoffs[np.nanargmin(log_rank_p)]
    
    # Save dataset to pickle file
    def save(self, filename=""):
        if filename == "":
            filename = "dataset.pkl"
        data = [self.biomarkers, self.biomarkers_mean, self.log_rank_p, self.cores_name]
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        
    # Load dataset from pickle file
    def load(self, filename=""):
        if filename == "":
            filename = "dataset.pkl"
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        loaded_biomarkers = data[0]
        loaded_biomarkers_mean = data[1]
        loaded_log_rank_p = data[2]
        loaded_cores_name = data[3]

        for key, value in loaded_biomarkers.items():
            self.biomarkers[key] = value
        for key, value in loaded_biomarkers_mean.items():
            self.biomarkers_mean[key] = value
        for key, value in loaded_log_rank_p.items():
            self.log_rank_p[key] = value
        if self.cores_name == []:
            self.cores_name = loaded_cores_name
