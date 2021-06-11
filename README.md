
Loop Q challenge
==============================

Project work for Loop Q Prize machine learning competition.

Running the notebooks
--------------------
Dependencies should be run before using the notebooks. You can install all dependencies, and check AWS connection by running
a bash script `initialize_project.sh` at root. 

If you want more control of your project initialization, you can use Makefile with `make` command. 
By typing `make`, you see list of options what you can do with it. With make you can install requirements, environment, and load data from the cloud.

Virtual environment is highly suggested!
to install virtual environment use (has been only tested with pyenv in ubuntu, not Conda).
```
make create_environment
```

To initialize requirements:
```
make requirements
```

To initialize the data from AWS:
```
make sync_data_from_s3
```


to check AWS connection and python version:
```
make test_environment
```

Loading the data and the model
------------------------------
Data and Torch model are saved in AWS S3 bucket, due to large size. Visitor credentials that have read permissions are shared privately to competition owners. 
These credentials should be included to project root in `.env` file for following fields:

```
AWS_DEFAULT_REGION=
AWS_ACCESS_KEY=
AWS_SECRET_ACCESS_KEY=
``` 


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   |── initialize_data.py
    |   |   |── preprocess_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── dimension_reduction.py
    │   │
    │   ├── models         <- Scripts to handle AWS util functions
    │        └── aws_utils.py                 
   


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
