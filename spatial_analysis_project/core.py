import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

class Core:
    def __init__(self, csv_file=None):
        if csv_file is not None:
            self.load_from_csv_(csv_file)

    # Load single cell map data from csv file
    def load_from_csv_(self, csv_file):
        self.df = pd.read_csv(csv_file)

        # Name of core and cell coordinates
        self.name = os.path.basename(csv_file).split(".")[0]
        self.cell_coordinates = self.df.loc[:, ["Centroid X px", "Centroid Y px"]].values / 0.497
        self.cell_areas = self.df.loc[:, ["Area px^2"]].values

        self.area = 0.2749**2 * np.pi * np.prod(np.max(self.cell_coordinates, axis=0) - np.min(self.cell_coordinates, axis=0))/4
    
        # Cell types and their integer encoding
        self.cell_types = self.df.loc[:, "Class Type"].values.tolist()
        self.cell_types_set = sorted(list(set(self.cell_types)))
        self.cell_types_dic = dict(zip(self.cell_types_set, np.arange(len(self.cell_types_set))))
        self.cell_types = np.array([self.cell_types_dic[cell] for cell in self.cell_types])

        self.cell_types_number = dict(zip(self.cell_types_set, [np.sum(self.cell_types == i) for i in np.arange(len(self.cell_types_set))]))
        self.cell_number = self.cell_types.shape[0]

        # dictionary for all biomarkers that are implemented
        self.biomarkers = {}

    # Load patient information from csv file
    def load_patient_from_csv_(self, csv_file):
        df = pd.read_csv(csv_file)

        # Check if core is contained in csv file
        if df.loc[df["Core ID"] == self.name].empty:
            raise ValueError("Patient information not found for core " + self.name) 
        
        # Set patient information
        self.patient_id = df.loc[df["Core ID"] == self.name]["Patient ID"].values[0]
        self.patient_months = int(df.loc[df["Core ID"] == self.name]["OS(Months)"].values[0])
        self.patient_status = bool(df.loc[df["Core ID"] == self.name]["Status"].values[0])

    # Calculate the distances between each of the cells in the core
    def calculate_cell_distances_(self):
        self.cell_distances = distance_matrix(self.cell_coordinates, self.cell_coordinates)
        self.cell_distances[self.cell_distances==0] = np.inf

    ################################
    ########## Biomarkers ##########
    ################################

    # Fraction of cells of a certain type in the core
    # One value for each cell type
    def fraction_cell_type(self):
        returns = {}
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_fraction"] = self.cell_types_number[cell_type] / self.cell_number
            returns[cell_type + "_fraction"] = self.biomarkers[cell_type + "_fraction"]
        return returns

    # Area density of cells of a certain type in the core
    # One value for each cell type
    def density_cell_type(self):
        returns = {}
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_density_mu^2"] = self.cell_types_number[cell_type] / self.area
            returns[cell_type + "_density_mu^2"] = self.biomarkers[cell_type + "_density_mu^2"]
        return returns
    
    # Average area of cells of a certain type in the core
    # One value for each cell type
    def area_cell_type(self):
        returns = {}
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_average_area_px^2"] = np.mean(self.cell_areas[self.cell_types == self.cell_types_dic[cell_type]])
            returns[cell_type + "_average_area_px^2"] = self.biomarkers[cell_type + "_average_area_px^2"]
        return returns
    
    # 1) For every cell of Type1, find all cells that are within radius
    # then, among those cells, calculate the average amount of cells of Type2
    # One value for each possible combination of Type1-Type2
    # 2) For every cell of Type1, find all cells that are within radius
    # then, count all these cells
    # One value for each type 
    def neighbouring_cell_type_distance_cutoff(self, radius=50):
        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        for cell_type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[cell_type]
            distances = self.cell_distances[mask, :]
            close_cells_idx = distances < radius
            counts = np.zeros(len(self.cell_types_set))

            for i in range(close_cells_idx.shape[0]):
                close_cells = self.cell_types[close_cells_idx[i,:]]
                counts += np.histogram(close_cells, bins=np.arange(len(self.cell_types_set)+1))[0]

            self.biomarkers["Average amount of cells within " + str(radius) + "mu around " + cell_type] = np.sum(counts) / np.sum(mask)
            returns["Average amount of cells within " + str(radius) + "mu around " + cell_type] = np.sum(counts) / np.sum(mask)

            for i in range(counts.shape[0]):
                self.biomarkers["Average amount of " + self.cell_types_set[i] + " cells within " + str(radius) + "mu of " + cell_type] = counts[i] / np.sum(mask)
                returns["Average amount of " + self.cell_types_set[i] + " cells within " + str(radius) + "mu of " + cell_type] = counts[i] / np.sum(mask)
        return returns
    
    # For every cell of Type1, find the k closest cells
    # then, among those cells, calculate the average fraction of cells of Type2
    # One value for each possible combination of Type1-Type2
    def neighbouring_cell_type_amount_cutoff(self, k=50):
        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()
        
        for cell_type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[cell_type]
            distances = self.cell_distances[mask, :]
            close_cells_idx = np.argpartition(distances, k, axis=1)[:, :k]
            counts = np.zeros(len(self.cell_types_set))

            for i in range(close_cells_idx.shape[0]):
                close_cells = self.cell_types[close_cells_idx[i,:]]
                counts += np.histogram(close_cells, bins=np.arange(len(self.cell_types_set)+1))[0]

            for i in range(counts.shape[0]):
                self.biomarkers["Fraction of " + self.cell_types_set[i] + " among " + str(k) + " closest cells next to " + cell_type] = counts[i] / (np.sum(mask)*k)
                returns["Fraction of " + self.cell_types_set[i] + " among " + str(k) + " closest cells next to " + cell_type] = counts[i] / (np.sum(mask)*k)
        return returns

    # For every cell of Type1, find the closest cell of Type2
    # then, calculate the average of all these distances
    # One value for each possible combination of Type1-Type2
    def smallest_distance_cell_type(self):
        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()
        
        for type1 in self.cell_types_set:
            for type2 in self.cell_types_set:
                mask1 = self.cell_types == self.cell_types_dic[type1]
                mask2 = self.cell_types == self.cell_types_dic[type2]
                distances = self.cell_distances[mask1, :]
                distances = distances[:, mask2]
                if distances.size == 0:
                    dist = np.nan
                else:
                    dist = np.mean(np.min(distances, axis=1))
                self.biomarkers["Average smallest distance from " + type1 + " to " + type2] = dist
                returns["Average smallest distance from " + type1 + " to " + type2] = dist
        return returns

    # For every cell of Type1, find the closest cell of Type2
    # then, calculate the cumulative distribution of all these distances (i.e. G function)
    # then, Calculate the difference between the empirical G function and the theoretical G function
    # One value for each possible combination of Type1-Type2
    def g_function(self):
        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        if not "Lymphocyte_density_mu^2" in self.biomarkers.keys():
            self.density_cell_type()

        for type1 in self.cell_types_set:
            for type2 in self.cell_types_set:
                mask1 = self.cell_types == self.cell_types_dic[type1]
                mask2 = self.cell_types == self.cell_types_dic[type2]
                distances = self.cell_distances[mask1, :]
                distances = distances[:, mask2]

                if distances.size == 0:
                    self.biomarkers["G-function for " + type1 + " to " + type2] = np.nan
                    returns["G-function for " + type1 + " to " + type2] = np.nan
                    continue

                distances = np.min(distances, axis=1)
                distances = np.sort(distances)        
                radius = np.linspace(0, 200, 100).reshape(-1, 1)
                g_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                g_function_theo = 1 - np.exp(-np.pi * radius**2 * (self.biomarkers[type2 + "_density_mu^2"]))
                g_diff = np.sum(g_function_theo - g_function_emp) / radius.shape[0]
                self.biomarkers["G-function for " + type1 + " to " + type2] = g_diff
                returns["G-function for " + type1 + " to " + type2] = g_diff

        return returns
