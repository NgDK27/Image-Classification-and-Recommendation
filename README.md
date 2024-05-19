# Image-Classification-and-Recommendation

#### Group SG_T3_G5
- S3800978	• Do Hoang Quan
- S3864235	• Lee Seungmin
- S3978724	• Nguyen Ich Kiet
- S3925921	• Nguyen Dinh Khai

--- 
## Setup
Our project uses `pdm` to manage dependencies and environments across Windows and macOS.

There's just some steps to take:
1. Install PDM
2. Install dependencies and create environment
3. Select the created environment as your interpreter
4. Copy in raw dataset


### Step 1. Install PDM

**For Linux/Mac**

```commandline
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

**For Windows**

```commandline
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | py -
```

### Step 2. Create dependencies and environment

**For Mac (tested) and Linux (untested)**

```commandline
pdm install -G macos
```

**For Windows**

```commandline
pdm install -G windows
```

### Step 3. Select the created environment as your Python interpreter

In your chosen IDE, select the `python.exe` within the newly created `.venv/Script/` folder as your Python interpreter.

### Step 4. Copy in raw dataset

Copy in the entire raw dataset folder `Furniture_Data` into `/data/raw/`. The final structure should look like this:
```
├── data/
│   └── raw/
│       └── Furniture_Data/
│           ├── beds/
│           │   ├── Asian/
│           │   │   └── ...
│           │   └── ... 
│           ├── chairs/
│           └── ...
└── ...
```

--- 
## Running

1. Ensure all steps in `Setup` is completed.
2. Run notebooks in `/notebooks` in the numerical order, OR, run notebook `0. Main. ipynb`, which will run all the notebooks for you. This is important as previous notebooks output files and models for later notebooks.
3. Once all the notebooks is completed, OR, at least `4. Recommend - Class` (read the `Notes` below for why), you can now run the front-end:
```commandline
cd streamlit
pdm run streamlit run app.py
```

**Note:**
- Task 1 and Task 3 models files are included.
- Task 2 systems are NOT included, as the recommendation system must be run from the start for good results. As such, please run at least `4. Recommend - Class` before starting the front-end.

---
## Project Structure

```
├── notebooks/
│   ├── 1. EDA.ipynb
│   ├── 2. Preprocessing.ipynb
│   ├── 3. Task 1 - ANN.ipynb
│   ├── 4. Task 1 - CNN.ipynb
│   ├── 5. Recommend - Class.ipynb
│   ├── 6. Task 3 - CNN.ipynb
│   └── 7. Recommend - Style.ipynb
├── streamlit/
│   ├── app.py
│   └── utils.py
├── utils/
│   ├── augmentation.py
│   ├── data.ipynb
│   ├── duplicates.py
│   ├── encoding.py
│   ├── tensorflow_preprocessing.ipynb
│   └── visualization.py
├── data/
│   └── raw/
│       └── Furniture_Data/
│           ├── beds/
│           │   ├── Asian/
│           │   │   └── ...
│           │   └── ... 
│           ├── chairs/
│           └── ...
├── .gitignore
├── app.py
├── pdm.lock
├── pyproject.toml
└── README.md
```

1. `notebooks/`: all Jupyter Notebooks for this project.
2. `data/`: contains original dataset. As you run the notebooks, more data will be added into this folder, such as:
-  `models/`: all the model in the project
-  `processed/`: all dataframe used (train, test, split, duplicates)
-  `recommend/`: recommendation dataframe and its feature vectors dataset
-  `raw/`: original dataset
3. `streamlit/`: contains code for running the Streamlit front-end.
4. `utils/`: contains various utility code to support the Jupyter notebooks.
5. `.gitignore`: contains ignore VCS ignore rules.
6. `app.py`: Streamlit front-end.
7. `pdm.lock`: latest dependencies lock to ensure reproducible environments.
8. `pyproject.toml`: required dependencies and Python version for PDM to install.
9. `README.md`: markdown file containing description of this project and how to run it.
