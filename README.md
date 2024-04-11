# Supervised Classification Model Training

This repository contains Python code for training supervised classification models using Support Vector Machine (SVM) and Random Forest algorithms. These models are trained on geospatial data, particularly orthophotos and corresponding shapefiles.

## Requirements
- Python 3.x
- Libraries:
    - rasterio
    - geopandas
    - tkinter
    - scikit-learn

## Usage
1. Clone or download this repository to your local machine.
2. Ensure you have the necessary Python libraries installed (see Requirements section).
3. Run the `main.py` file.
4. The GUI window will prompt you to provide paths to orthophotos and shapefiles, as well as specify parameters for model training (test size and choice of algorithm).
5. After providing the required inputs and clicking the "OK" button, the code will start processing the data, training the selected model, and saving it to disk.

## File Structure
- `main.py`: Main Python script containing the code for model training.
- `README.md`: This README file providing an overview of the project.
- `model_svm_[filename].pkl`: Saved SVM model files.
- `model_rf_[filename].pkl`: Saved Random Forest model files.

## Contributions
Contributions to this project are welcome. If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
