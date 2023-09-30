# AC-DL-LAB-SS-2022-Team03

## General Information
The project is tackling the interesting problem presented by the HAPT (Human Activity Recognition Dataset).
Please find documentation and detailed description of the dataset and it's feature set, target set at the [following link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

  - _dataset_ is NOT added to the project on GitHub, please link it to the Google Colab instance + add path in [dataset.py](https://github.tik.uni-stuttgart.de/ac138165/AC-DL-LAB-SS-2022-Team03/blob/master/dataset.py)
  - _models_ are NOT added in the project on GitHub, since these will be presented using the paper & presentation & poster
  - _training_ metrics are NOT added in the project on GitHub, since these are used as presentation materials for the publications mentioned above

## Running the project
Main project entry point: [project-master.ipynb](https://github.tik.uni-stuttgart.de/ac138165/AC-DL-LAB-SS-2022-Team03/blob/master/project-master.ipynb)
### Prerequisites:
  - project is created to be executed using a Jupiter instance, preferably Google Colab
  - prerequisites are added in the **Setup** cell tree in the _ipynb_ file

### Running parts of the project:
<details>
<summary>opening the _ipynb_ file, and running the <b>Setup</b> cell</summary>
<br>

  - Google Drive will be linked, used to link dataset to project
  - needed Python3 packages (besides already installed ones on Colab) are installed
  - _cwd_ is set accordingly to the path of project on Google Drive
</details>
  
<details>
<summary>the <b>validation cells for source code</b> cell tree contains different cells to check</summary>
<br>

  - dataset loading
  - configuration variables such as device (CPU|GPU)
</details>

<details>
<summary>the <b>train-val-test sequce</b> is the main entry point for creating models</summary>
<br>

  - open [main.py](https://github.tik.uni-stuttgart.de/ac138165/AC-DL-LAB-SS-2022-Team03/blob/master/main.py)
  - modify in function main() the model you would like to train/evaluate
    - uncomment in model declaration
    - for RNNs, DO NOT forget sequence_length variable, also decomment it if needed
    - Linear networks: sequence_length has to be None
	  
  - execute cells from the cell tree, for the following purposes:
    - train the currently active (decommented in main.py) model
	- visualize losses throughout training (training and validation) - in order to assess correct combatting of overfit
    - test the currently active model
    - explain the current model using SHAP: average feature importance & one index in test dataset explained using force plots
</details>

<details>
<summary>the <b>(Empirical) Evaluation - model comparison</b> cell tree is for model comparison, evaluation</summary>
<br>

  - run the first cell to see training metrics **loss|acc** by specifying the train_cmp parameter, and add a number as sg_w parameter to smoothen the metrics (good smoothening is 1001, very visible differences between metrics)
  - the second cell compares the evaluation of every existing trained model by printing
    - (optional, param show_arch) the architecture of the network, together with tensor sizes, network sizes
    - (optional, param conf_cut) the size of the subset from test dataset to be shown on confusion matrix + sample/category from subset (500 is a good value to visually evaluate)
    - evaluation metrics such as accuracy, f1
</details>

## TODO List
### Done:
  - train-validation-test sequence
  - SHAP (SHapley Additive exPlanations) analysis of trained models
  - empirical evaluation
    - visualization of training metrics as comparison plots
	- model evaluation comparison. For each model:
	  - Model architecture
	  - Metrics: training accuracy, F1 score
	  - Visualize: confusion matrix on a subset of the test dataset (understand metrics)
  - add EarlyStopping to training, together with saving loss (train/val ds) + val acc **per epoch**
  - enhance RNN architecture to capture better the task
  - work on GRU/LSTM comparison in particular
  - use SHAP to explain different scenarios (idea: explain how e.g. sitting is affected differently than walking upstairs)
  - use multiple sizes/network architecture cluster to visualize problem's complexity
		
### TODO:
  - use further researched networks to prove different theoretical aspects (transformers, ensemble methods, etc.)
  
