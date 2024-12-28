# README.md

This README provides detailed instructions for running the scripts in the correct order to ensure the successful execution of the **MHCD Framework** pipeline. Each script's functionality is briefly explained to guide users through the process.

---

## Prerequisites

Before running the scripts, ensure you have the following:
- Python (version >= 3.8)
- Required Python libraries (install using `pip install -r requirements.txt` if provided)
- Input datasets (e.g., PISA data files or related inputs)

---

## Execution Order

Follow the steps below to execute the scripts in sequence:

### Step 1: Build the Q-Matrix
**Script:** `1-build_Q_matrix.py`

- Description: This script generates the Q-Matrix, which maps exercises to their associated knowledge concepts (KCs).
- Output: A Q-Matrix file.

### Step 2: Transform PISA Data
**Script:** `2-pisa-transform_to_number.py`

- Description: Converts the PISA dataset's categorical values into numerical format for further processing.
- Output: Transformed numerical dataset.

### Step 3: Compute Average Scores
#### Step 3.1: Compute Student Exercise Average Score
**Script:** `3.1-Compute student exercies average socre and Save_log_data_json.py`

- Description: Calculates the average score of exercises for each student and saves the data in a JSON format.
- Output: Log data in JSON format.

#### Step 3.2: Compute KC Average Score
**Script:** `3.2-Compute KC Average score.py`

- Description: Calculates the average score for each knowledge concept (KC).
- Output: Average scores per KC.

### Step 4: One-Hot Encode Scores
**Script:** `4-One-hot-encoded-socres.py`

- Description: Converts scores into one-hot encoded representations for subsequent modeling.
- Output: One-hot encoded score files.

### Step 5: Divide Data
**Script:** `5-divide_data.py`

- Description: Divides the processed data into training, validation, and testing sets.
- Output: Training, validation, and test datasets.

### Step 6: Build Knowledge-Exercise Graphs
#### Step 6.1: Build K-E Graph
**Script:** `6.1-build_k_e_graph.py`

- Description: Constructs a graph representing the relationships between knowledge concepts and exercises.
- Output: K-E graph structure.

#### Step 6.2: Build K-E Hierarchy Graph
**Script:** `6.2-build_k_e_hierarchy_graph.py`

- Description: Extends the K-E graph to include hierarchical relationships among knowledge concepts and exercises.
- Output: Hierarchical K-E graph structure.

### Step 7: Build User-Exercise Graphs
#### Step 7.1: Build U-E Graph
**Script:** `7.1-build_u_e_graph.py`

- Description: Creates a graph representing the relationships between users (students) and exercises.
- Output: U-E graph structure.

#### Step 7.2: Build U-E Hierarchy Graph
**Script:** `7.2-build_u_e_hierarchy_graph.py`

- Description: Extends the U-E graph to include hierarchical relationships among users and exercises.
- Output: Hierarchical U-E graph structure.

### Step 8: Train the MHCD Model
**Script:** `train_MH.py`

- Description: Trains the Multi-Hierarchy Interactive Constraint-aware Cognitive Diagnosis (MHCD) model using the preprocessed data and generated graphs.
- Output: Trained MHCD model.

---

## Notes
- Ensure all required input files are correctly placed in their respective directories before running the scripts.
- For any errors or issues, refer to the comments in each script or contact the author.
- If available, check `requirements.txt` for additional dependencies and install them using:
  ```bash
  pip install -r requirements.txt
  ```



