# ss-py

These are two simple python scripts to create simple stuff, location and pitching models based on baseballsavant pitching data. You can browse through the results [here](https://enzdor.github.io/stuff-plus-simple). There are a total of nine models, three for each of the already mentioned above. These three are for the three different pitch type groups: fastballs, offspeeds, and breaking-balls. So, for example, the stuff models are stuff_fastball, stuff_breaking, and stuff_offspeed. The same is true for location and pitching.

The models are created using [FLAML](https://github.com/microsoft/FLAML) and using xgboost as the estimator. You can check all of the regressors used for the different models in the code. You can find more information on how this works by reading through the code and its comments and by going to this small [post](https://enzdor.github.io/stuff-plus-simple/how.html) which also contains some small analysis of the models and their resutls. You can also run `python predict.py --help` and `python ml_stuff.py --help`.

## Usage

Before you run everything, you need to instll the dependecies:

```
pip install flaml xgboost scikit-learn pandas numpy
```

Now, you need to have the data from [baseballsavant](https://baseballsavant.mlb.com). You can download it from the search tab or using a script. The script is use is [dl-bs](https://github.com/enzdor/dl-bs) but there are other options like [pybaseball](https://github.com/jldbc/pybaseball) and [baseballr](https://github.com/billpetti/baseballr). The data should be in csv format and the format should be exactly the same as downloading directly from baseballsavant.

If you want to create your own models:

```
python3 ml_stuff.py yourdata.csv
```

You can change the budget time used for training the models in `ml_stuff.py` in the automl config. You can also add regressors or remove them. The output given by running the above script is a log file containing information about the models and a folder called `models` were all the models are stored in pickle format.

If you want to make predictions with the models inside the `models` folder you need to:

```
python predict.py yourdata.csv
```

The output will be a database in sqlite format containing all of the results. If there already exists a database called `predict.db` you should either change its name or use `python predict.py -odb="mydb.db" yourdata.csv`. Always remember to chec if there already exists a database with the same its going to be output, if you don't change anything, the program will fail.

## TODO

- add script to create all graphs with matplotlib
- export per season means with predict.py 
