# CSCI-6502

This repository contains the source code for the final project of **CSCI 6502**. The project is organized into two main folders with Python scripts used throughout the data processing and training pipeline. Additionally, the notebook titled **`ResultsPreparation.ipynb`** demonstrates how the figures presented in the final paper were generated.

Some figures were also captured directly from TensorBoard as screenshots. However, the full TensorBoard profiling files—each approximately **1 GB** in size—were not uploaded due to storage constraints. If you have any questions or would like access to these files, feel free to contact me.

---

## Project Structure

### `DataPreparation/`
This folder contains the scripts responsible for collecting and preprocessing the data:

- All scripts **except** `CombineData.py` are used to scrape data from Yahoo Finance and convert it into spectrograms.
- Each script includes a clearly defined set of global variables at the top, allowing users to configure how data is collected and transformed.
- Once all spectrograms have been generated, run `CombineData.py` to concatenate the data into arrays for use in training and evaluation.

### `Training/`
This folder contains the training scripts used to build and evaluate models. These scripts were executed on a configured VM hosted on **Google Cloud Platform (GCP)**.

Key scripts include:

- `TrainingPyTorchDP.py`  
- `TrainingTensorFlowMS.py`  
  These two scripts generated the majority of results used in the final report and visualizations.

- `TrainingPyTorchDPParallel.py`  
  A variation of `TrainingPyTorchDP.py` that enables parallel data loading using `num_workers` and `pin_memory` in PyTorch's DataLoader. Used to explore performance improvements via parallel data processing.

- `TrainingTensorFlowMSProfile.py`  
  A variant of `TrainingTensorFlowMS.py` that includes a callback for detailed epoch-level profiling. Useful for in-depth performance analysis.

---

## Results

The **`ResultsPreparation.ipynb`** notebook outlines how the visualizations in the final report were generated from the experimental data.

---

## Contact

If you have any questions or would like access to the TensorBoard profiling files, please feel free to reach out.

