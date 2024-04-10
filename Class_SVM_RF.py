# Created by @Rigles Maia
# Completion Date: 06/02/2024
# Routine for training supervised classification models Support Vector Machine and Random Forest

####################################***************** - Creating the GUI - *****************####################################
import rasterio
import rasterio.mask
import joblib
import numpy as np
import geopandas as gpd
import tkinter as tk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os

# Function to get values from the GUI
def get_values():
    # Function called when the "OK" button is clicked
    global ortho_path, shape_path, TZ, model
    ortho_path = str(entry_ortho_path.get())
    shape_path = str(entry_shape_path.get())
    TZ = float(entry_TZ.get())
    model = int(entry_model.get())
   
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Parameter Configuration")
root.geometry("400x550")  # Adjusting window size

# Change font size
font = ("Helvetica", 15)

# Add input fields and labels
label_ortho_path = tk.Label(root, font=font, text="Orthophotos:")
label_ortho_path.pack()
entry_ortho_path = tk.Entry(root, font=font)
entry_ortho_path.pack()

label_shape_path = tk.Label(root, font=font, text="Shapes:")
label_shape_path.pack()
entry_shape_path = tk.Entry(root, font=font)
entry_shape_path.pack()

label_TZ = tk.Label(root, font=font, text="Test Size:")
label_TZ.pack()
entry_TZ = tk.Entry(root, font=font)
entry_TZ.pack()

label_model = tk.Label(root, font=font, text="SVM or RF (1 and 2):")
label_model.pack()
entry_model = tk.Entry(root, font=font)
entry_model.pack()

# Add OK button
button_ok = tk.Button(root, text="OK", font=font, command=get_values)
button_ok.pack()

# Start GUI loop
root.mainloop()

####################################***************** - Model Training - *****************####################################

def train_and_save_model(tiff_file, model, X_train, X_test, y_train, y_test):
    if model == 1:
        print("***********************Chosen Model Support Vector Machine***********************")
        # Define the SVM model
        model = SVC(random_state=42)

        # Hyperparameter search space
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': ['scale'],
                      'kernel': ['linear', 'rbf','poly', 'sigmoid']
                      }

        # Create GridSearchCV
        grid = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1, refit=True)
        
        # Train the model
        grid.fit(X_train, y_train)

        # Print best parameters
        print(f"Best Parameters for {tiff_file}: {grid.best_params_}")

        # Predictions and Evaluation
        y_pred = grid.predict(X_test)
        print(f"Accuracy for {tiff_file}: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

        # Save the model
        model_name = f'svm_model_{tiff_file.replace(".tif", "")}.pkl'
        joblib.dump(grid.best_estimator_, model_name)

    elif model == 2:
        print("***********************Chosen Model Random Forest***********************")
        # Define the Random Forest model
        model = RandomForestClassifier(random_state=42)

        # Hyperparameter search space
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300, 500],
            'max_features': ['sqrt'],
            'max_depth': [1, 3, 5, 7, 11]
        }

        # Create GridSearchCV
        grid = GridSearchCV(model, param_grid, cv=5, verbose=3, n_jobs=-1, refit=True)

        # Train the model
        grid.fit(X_train, y_train)

        # Print best parameters
        print(f"Best Parameters for {tiff_file}: {grid.best_params_}")

        # Predictions and Evaluation
        y_pred = grid.predict(X_test)
        print(f"Accuracy for {tiff_file}: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

        # Save the model
        model_name = f'rf_model_{tiff_file.replace(".tif", "")}.pkl'
        joblib.dump(grid.best_estimator_, model_name)

    else:
        print("Unrecognized model. Please choose 1 for SVM or 2 for Random Forest.")

####################################***************** - Labeling and Data Splitting - *****************####################################

# Paths to folders containing TIFF images and shapefiles
tiff_folder_path = ortho_path
shape_folder_path = shape_path

# List all TIFF files and shapefiles
tiff_files = [f for f in os.listdir(tiff_folder_path) if f.endswith('.tif')]
shape_files = [f for f in os.listdir(shape_folder_path) if f.endswith('.shp')]

# Ensure the number of TIFF files and shapefiles is equal
assert len(tiff_files) == len(shape_files), "Unequal number of TIFF files and shapefiles."

# Process each pair of files
for tiff_file, shape_file in zip(tiff_files, shape_files):
    tiff_path = os.path.join(tiff_folder_path, tiff_file)
    shape_path = os.path.join(shape_folder_path, shape_file)

    # Load the shapefile
    gdf = gpd.read_file(shape_path)

    # Load the image and extract features and labels
    X = []  # Features
    y = []  # Labels

    with rasterio.open(tiff_path) as src:
        for _, row in gdf.iterrows():
            out_image, out_transform = rasterio.mask.mask(src, [row['geometry']], crop=True)
            if out_image.size == 0:
                continue

            n_bands, n_rows, n_cols = out_image.shape
            out_image_reshaped = out_image.reshape(n_bands, n_rows * n_cols).T
            X.extend(out_image_reshaped)
            y.extend([row['id']] * out_image_reshaped.shape[0])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TZ, random_state=42)
    
    # Control the code and call training functions repeatedly within the loop
    train_and_save_model(tiff_file, model, X_train, X_test, y_train, y_test)
