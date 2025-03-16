import pandas as pd
import numpy as np

# Make sure the clean1.data file is in the same directory as where this function is called

def clean_musk():

    file_path = 'clean1.data'
    
    original_data_df = pd.read_csv(file_path, header=None)

    # Extracting the molecule names, features, and labels
    molecule_names = original_data_df.iloc[:, 0]

    # Drop the first 2 columns (molecule_name and conformation_name)
    features = original_data_df.iloc[:, 2:-1]  
    labels = original_data_df.iloc[:, -1]

    # Initializing lists to store feature arrays and labels for each molecule
    feature_arrays = []
    label_arrays = []

    # Grouping by molecule_name
    for name, group in features.groupby(molecule_names):
        # Extracting the feature array for this molecule
        feature_array = group.to_numpy(dtype=int)
        feature_arrays.append(feature_array)

        # Extracting the corresponding labels for the conformations of this molecule
        # The label for the molecule (bag) will be '1' (musk) if any of its conformations is labeled '1'
        labels_array = labels[molecule_names == name].to_numpy()
        molecule_label = 1 if 1 in labels_array else 0
        label_arrays.append(molecule_label)

    return feature_arrays, np.array(label_arrays)

if __name__ == "__main__":
    print(clean_musk()[0][0])
    print(clean_musk()[1][0])