import numpy as np
import pandas as pd
import sys
import os
import argparse
import pickle
import math
from flaml import AutoML
from sklearn.model_selection import train_test_split

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        usage: python ml_stuff.py infile.csv [-h] 
               [-osum sum.txt] [-m ./models-folder]

        Script to get models in pickle format. The input data 
        should be in csv format. The output is a foldercontaining
        all of the models in pickle format. To predict data see
        predict.py .
""")

parser.add_argument("input_file_path")
parser.add_argument("-osum", "--outfile-summary", dest="outfile_summary", default="models.log", help="""
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

y0 = 50
yf = 17/12

df['vy_f'] = [-math.sqrt(x * x - (2 * y * (y0 - yf))) for x, y in zip(df['vy0'], df['ay'])]
df['t'] = [(x - y) / z for x, y, z in zip(df['vy_f'], df['vy0'], df['ay'])]
df['vz_f'] = [x + (y * z) for x, y, z in zip(df['vz0'], df['az'], df['t'])]
df['vx_f'] = [x + (y * z) for x, y, z in zip(df['vx0'], df['ax'], df['t'])]

# vertical and horizontal approach angle

df['vaa'] = [math.degrees(-math.atan(x / y)) for x, y in zip(df['vz_f'], df['vy_f'])]
df['haa'] = [math.degrees(-math.atan(x / y)) for x, y in zip(df['vx_f'], df['vy_f'])]

# trying to find non magnus movement

baseball_radius = 2.94 / 2

df['spin_factor'] = [baseball_radius * x / y for x, y in zip(df['release_spin_rate'], df['vy0'])]
df['lift_coefficient'] = [0.366 * (1 - math.exp(-6.041 * x[0])) for x in zip(df['spin_factor'])]

df['total_transverse_mov_x'] = [1/2 * x * pow(y, 2) for x, y in zip(df['ax'], df['t'])]
df['total_transverse_mov_z'] = [1/2 * x * pow(y, 2) for x, y in zip(df['az'], df['t'])]
df['total_transverse_dir'] = [
    (math.degrees(math.atan2(x, z)) + 360) % 360
    for x, z in zip(df['total_transverse_mov_x'], df['total_transverse_mov_z'])
]

k_factor = 0.00544

df['release_spin_rate_x'] = [x * math.sin(math.radians(y)) for x, y in zip(df['release_spin_rate'], df['spin_axis'])]
df['release_spin_rate_z'] = [x * math.cos(math.radians(y)) for x, y in zip(df['release_spin_rate'], df['spin_axis'])]

# df['magnus_transverse_dir'] = [math.degrees(-math.atan(x / y)) for x, y in zip(df['release_spin_rate_x'], df['release_spin_rate_z'])]
df['magnus_transverse_dir'] = [(math.degrees(math.atan2(x, z)) + 360) % 360 for x, z in zip(df['release_spin_rate_x'], df['release_spin_rate_z'])]

df['magnus_ax'] = [x * k_factor * pow(y, 2) * math.cos(math.radians(z)) for x, y, z in zip(df['lift_coefficient'], df['vy0'], df['magnus_transverse_dir'])]
df['magnus_az'] = [x * k_factor * pow(y, 2) * math.sin(math.radians(z)) for x, y, z in zip(df['lift_coefficient'], df['vy0'], df['magnus_transverse_dir'])]

df['axis_differential'] = [x - y for x, y in zip(df['total_transverse_dir'], df['magnus_transverse_dir'])]
df['abs_axis_differential'] = [abs(x[0]) for x in zip(df['axis_differential'])]

# efficient spin is only on the transverse axis, not gyrospin
# df['spin_efficiency_approx'] = [(math.sqrt(pow(x, 2) + pow(y, 2))) / z for x, y, z in zip(df['release_spin_rate_z'], df['release_spin_rate_x'], df['release_spin_rate'])]

# coors
df['coors'] = [1 if x[0] == "COL" else 0 for x in zip(df['home_team'])]

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

# for pitching location and stuff
for tc in to_calculate:
    summary_s += ("\n" + tc + "\n")
    dfs = df_groups

    # for each pitch type group
    for l in range(len(dfs)):
        summary_s += (categories[l] + "\n")
        regs_stuff = ['release_extension', 'spin_axis', 'release_spin_rate', 'az', 'ay', 
                    'ax', 'vz0', 'vy0', 'vx0', 'pfx_z', 'pfx_x', 'release_pos_z', 
                    'release_pos_y', 'release_pos_x', 'release_speed', 'vaa', 'haa',
                    'abs_axis_differential', 'coors']
        regs_location = ['c32', 'c22', 'c12', 'c02', 'c31', 'c21', 'c11', 'c01', 'c30', 
                    'c20', 'c10', 'c00', 'plate_z', 'plate_x', ]

        regressor_ml = []

        if tc == "stuff":
            regs = regs_stuff
            for r in regs:
                regressor_ml.append(dfs[l][r].to_list())

        elif tc == "location":
            regs = regs_location
            for r in regs:
                regressor_ml.append(dfs[l][r].to_list())

        elif tc == "pitching":
            regs = regs_stuff + regs_location
            for r in regs:
                regressor_ml.append(dfs[l][r].to_list())

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


        # automl get models


    #################################################

        automl = AutoML()

        automl_settings = {
            "time_budget" : 3600 * 2,
            "metric" : "mse",
            "task" : "regression",
            "log_file_name" : "ml_stuff.log",
            "estimator_list": ["xgboost"],
        }

        automl.fit(X_train, y_train, **automl_settings)
        r2 = automl.score(X_train, y_train)
        r2_test = automl.score(X_test, y_test)

    #################################################


        # write to summary variable important model 
        # info


    #################################################

        summary_s += str(automl.model.estimator) + "\n"
        summary_s += str(automl.model.estimator.feature_importances_) + "\n"
        summary_s += str(automl.feature_names_in_) + "\n"
        summary_s += str(automl.best_config) + "\n"
        summary_s += str(automl.best_loss) + "\n"
        summary_s += str(automl.best_loss_per_estimator) + "\n"
        summary_s += str(automl.metrics_for_best_config) + "\n"
        summary_s += str(r2) + " r2 \n"
        summary_s += str(r2_test) + " r2 test\n\n"

    #################################################


        # save models in path to folder given by
        # user


    #################################################

        os.makedirs(args.models_path, exist_ok=True)
        with open(f"{args.models_path}/{tc}_{categories[l]}.pkl", "wb") as f:
            pickle.dump(automl.model, f)

#################################################


    # write summary file


#################################################

with open(args.outfile_summary, "w") as text_file:
    text_file.write(summary_s)

