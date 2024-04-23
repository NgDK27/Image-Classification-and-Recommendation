# Image-Classification-and-Recommendation

## Project Structure


```
├── notebooks/
│   ├── 1.EDA.ipynb
│   ├── 2.Data_Processing_.ipynb
│   └── 3.Model_Building.ipynb
│   └── 4. Optimization.ipynb
├── data/
│   └── result/
│   ├── code_book.txt
│   ├── data_test.csv
│   └── data_train.csv
│   └── cleaned_test.csv
│   ├── cleaned_train.csv
│   ├── transformed_test.csv
│   └── transformed_train.csv
├── images/
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

1. `notebooks/`: This folder contains all Jupyter Notebooks for this project.
2. `data/`: This folder contains the original dataset, the cleaned and preprocessed one as well as the data description and the prediction result. 
3. `images/`: Contains the plots.
4. `requirements.txt`: Text file for `pip` installation of necessary packages for development environment.
5. `.gitignore`: This file contains ignore VCS ignore rules.
6. `README.md`: A text file containing useful reference information about this project, including how to run the algorithm.
7. `LICENSE`: MIT

---


## How to run

<span style="color:gold">**NOTE:**</span> It is recommended that you use a Python virtual environment to avoid conflict with your global packages, and to keep your global Python installation clean. 

On your terminal, make sure that you have the environment activated for the client script to have access to all required packages:

- Python Virtualenv:

   ```bash
   ./venv/Scripts/activate
   ```

- Conda:

   ```bash
   conda activate ./envs
   ```


```bash
pip install -r requirements.txt
```

Refer to [requirements.txt](requirements.txt) for package dependencies and their versions.

Then run from the first notebook