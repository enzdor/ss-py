import numpy as np
import pandas as pd
import sys
import os
import argparse
import pickle
from flaml import AutoML
from sklearn.model_selection import train_test_split

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        usage: python ml_stuff.py infile.csv [-h] [-odb data.db]
               [-osum sum.txt] [-m ./models-folder]

        Script to get models in pickle format. The input data 
        should be in csv format. The output is a foldercontaining
        all of the models in pickle format. To predict data see
        predict.py .
""")

parser.add_argument("input_file_path")
parser.add_argument("-osum", "--outfile-summary", dest="outfile_summary", default="ml_stuff_sum.log", help="""
        The path for the summary, a text file.
""")
parser.add_argument("-m", "--models", dest="models_path", default="./models", help="""
        The path for models folder, default is ./models .
""")

args = parser.parse_args()

if len(sys.argv) < 2:
    print("""
    There was no path provided for the baseball savant data.
    Run 
          python ml_stuff.py path_to_bs_data.csv
    or
          python ml_stuff.py --help
          """)
    quit()

if len(args.input_file_path) < 1:
    print("There was no path provided for the baseball savant data.")
    quit()

df = pd.read_csv(args.input_file_path)

if len(df) < 1:
    print("The csv does not contain any rows.")
    quit()

#################################################


    # creating new variables


#################################################

# string for the final summary file
summary_s = ""

fastballs = ["FF", "SI"]
offspeeds = ["FC", "CH", "FS", "FO", "SC"]
breaking_balls = ["CU", "KC", "CS", "SL", "ST", "SV", "KN"]
categories = ["fastball", "offspeed", "breaking_ball"]
categories_types = [fastballs, offspeeds, breaking_balls]

# 1 is true

category = []

for i in df['pitch_type'].to_list():
    if i in fastballs:
        category.append("fastball")
    elif i in offspeeds:
        category.append("offspeed")
    elif i in breaking_balls:
        category.append("breaking_ball")
    else:
        category.append("none")

df['category'] = category
df['c00'] = [1 if (x == 0 and y == 0) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c10'] = [1 if (x == 1 and y == 0) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c20'] = [1 if (x == 2 and y == 0) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c30'] = [1 if (x == 3 and y == 0) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c01'] = [1 if (x == 0 and y == 1) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c11'] = [1 if (x == 1 and y == 1) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c21'] = [1 if (x == 2 and y == 1) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c31'] = [1 if (x == 3 and y == 1) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c02'] = [1 if (x == 0 and y == 2) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c12'] = [1 if (x == 1 and y == 2) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c22'] = [1 if (x == 2 and y == 2) else 0 for x, y in zip(df['balls'], df['strikes'])]
df['c32'] = [1 if (x == 3 and y == 2) else 0 for x, y in zip(df['balls'], df['strikes'])]


#################################################


    # separate into three different categories


#################################################

df_f = df[df['category'] == "fastball"]
df_o = df[df['category'] == "offspeed"]
df_b = df[df['category'] == "breaking_ball"]

df_f = df_f.reset_index()
df_o = df_o.reset_index()
df_b = df_b.reset_index()

df_groups = [df_f, df_o, df_b]
to_calculate = ["pitching", "location", "stuff"]

#################################################


    # get regressors


#################################################

for tc in to_calculate:
    summary_s += ("\n" + tc + "\n")
    dfs = df_groups

    for l in range(len(dfs)):
        summary_s += (categories[l] + "\n")

        regressor_ml = []

        if tc == "stuff":
            regressor_ml.append(dfs[l]['release_speed'].to_list())
            regressor_ml.append(dfs[l]['release_pos_x'].to_list())
            regressor_ml.append(dfs[l]['release_pos_y'].to_list())
            regressor_ml.append(dfs[l]['release_pos_z'].to_list())
            regressor_ml.append(dfs[l]['pfx_x'].to_list())
            regressor_ml.append(dfs[l]['pfx_z'].to_list())
            regressor_ml.append(dfs[l]['vx0'].to_list())
            regressor_ml.append(dfs[l]['vy0'].to_list())
            regressor_ml.append(dfs[l]['vz0'].to_list())
            regressor_ml.append(dfs[l]['ax'].to_list())
            regressor_ml.append(dfs[l]['ay'].to_list())
            regressor_ml.append(dfs[l]['az'].to_list())
            regressor_ml.append(dfs[l]['release_spin_rate'].to_list())
            regressor_ml.append(dfs[l]['spin_axis'].to_list())
            regressor_ml.append(dfs[l]['release_extension'].to_list())
        elif tc == "location":
            regressor_ml.append(dfs[l]['plate_x'].to_list())
            regressor_ml.append(dfs[l]['plate_z'].to_list())
            regressor_ml.append(dfs[l]['c00'].to_list())
            regressor_ml.append(dfs[l]['c10'].to_list())
            regressor_ml.append(dfs[l]['c20'].to_list())
            regressor_ml.append(dfs[l]['c30'].to_list())
            regressor_ml.append(dfs[l]['c01'].to_list())
            regressor_ml.append(dfs[l]['c11'].to_list())
            regressor_ml.append(dfs[l]['c21'].to_list())
            regressor_ml.append(dfs[l]['c31'].to_list())
            regressor_ml.append(dfs[l]['c02'].to_list())
            regressor_ml.append(dfs[l]['c12'].to_list())
            regressor_ml.append(dfs[l]['c22'].to_list())
            regressor_ml.append(dfs[l]['c32'].to_list())
        elif tc == "pitching":
            regressor_ml.append(dfs[l]['release_speed'].to_list())
            regressor_ml.append(dfs[l]['release_pos_x'].to_list())
            regressor_ml.append(dfs[l]['release_pos_y'].to_list())
            regressor_ml.append(dfs[l]['release_pos_z'].to_list())
            regressor_ml.append(dfs[l]['pfx_x'].to_list())
            regressor_ml.append(dfs[l]['pfx_z'].to_list())
            regressor_ml.append(dfs[l]['vx0'].to_list())
            regressor_ml.append(dfs[l]['vy0'].to_list())
            regressor_ml.append(dfs[l]['vz0'].to_list())
            regressor_ml.append(dfs[l]['ax'].to_list())
            regressor_ml.append(dfs[l]['ay'].to_list())
            regressor_ml.append(dfs[l]['az'].to_list())
            regressor_ml.append(dfs[l]['release_spin_rate'].to_list())
            regressor_ml.append(dfs[l]['spin_axis'].to_list())
            regressor_ml.append(dfs[l]['release_extension'].to_list())
            regressor_ml.append(dfs[l]['plate_x'].to_list())
            regressor_ml.append(dfs[l]['plate_z'].to_list())
            regressor_ml.append(dfs[l]['c00'].to_list())
            regressor_ml.append(dfs[l]['c10'].to_list())
            regressor_ml.append(dfs[l]['c20'].to_list())
            regressor_ml.append(dfs[l]['c30'].to_list())
            regressor_ml.append(dfs[l]['c01'].to_list())
            regressor_ml.append(dfs[l]['c11'].to_list())
            regressor_ml.append(dfs[l]['c21'].to_list())
            regressor_ml.append(dfs[l]['c31'].to_list())
            regressor_ml.append(dfs[l]['c02'].to_list())
            regressor_ml.append(dfs[l]['c12'].to_list())
            regressor_ml.append(dfs[l]['c22'].to_list())
            regressor_ml.append(dfs[l]['c32'].to_list())

        X = []
        y = dfs[l]['delta_run_exp']

        for i in range(len(y)):
            obs = []
            for x in regressor_ml:
                obs.append(x[i])
            X.append(obs)

    #################################################


        # drop observations with nans


    #################################################

        to_drop = []

        for i in np.argwhere(np.isnan(np.array(y))):
            if i[0] not in to_drop:
                to_drop.append(i[0])

        for i in np.argwhere(np.isnan(np.array(X))):
            if i[0] not in to_drop:
                to_drop.append(i[0])

        to_drop = list(reversed(sorted(to_drop)))

        for i in to_drop:
            X.pop(i)
            y.pop(i)
            category.pop(i)
            dfs[l] = dfs[l].drop(i)

        summary_s += ("Observations dropped due to missing values:" + str(len(to_drop)) + "\n")

    #################################################


        # split test and training set 


    #################################################

        X_raw = np.array(X)
        y_train = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X_raw, y_train, test_size=0.2, random_state=11)

    #################################################


        # automl


    #################################################

        automl = AutoML()

        automl_settings = {
            "time_budget" : 1,
            "metric" : "r2",
            "task" : "regression",
            "log_file_name" : "ml_stuff.log",
            "estimator_list": ["xgboost"],
        }

        automl.fit(X_train, y_train, **automl_settings)
        r2 = automl.score(X_train, y_train)
        r2_test = automl.score(X_test, y_test)

        summary_s += str(automl.model.estimator) + "\n"
        summary_s += str(automl.model.estimator.feature_importances_) + "\n"
        summary_s += str(automl.best_config) + "\n"
        summary_s += str(automl.best_loss) + "\n"
        summary_s += str(automl.best_loss_per_estimator) + "\n"
        summary_s += str(automl.metrics_for_best_config) + "\n"
        summary_s += str(r2) + " r2 \n"
        summary_s += str(r2_test) + " r2 test\n\n"

        # expected delta run expectancy
        dfs[l]['x_rv'] = automl.predict(X_raw)

        os.makedirs(args.models_path, exist_ok=True)
        with open(f"{args.models_path}/{tc}_{categories[l]}.pkl", "wb") as f:
            pickle.dump(automl.model, f)


with open(args.outfile_summary, "w") as text_file:
    text_file.write(summary_s)

