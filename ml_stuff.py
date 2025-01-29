import numpy as np
import datetime as dt
import pandas as pd
import sqlite3
import sys
import argparse
from flaml import AutoML

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        usage: python ml_stuff.py infile.csv [-h] [-odb data.db]
               [-osum sum.txt]

        Script to calculate stuff, location and pitching plus from
        baseball savant data. The input data should in csv format.
        The output is a sqlite db containing the resulting stats
        for each pitcher included in the original csv.

""")

parser.add_argument("input_file_path")
parser.add_argument("-odb", "--outfile-db", dest="outfile_db", default="important.db", help="""
        The path for the outfile, the sqlite db with the results.
""")
parser.add_argument("-osum", "--outfile-summary", dest="outfile_summary", default="important.txt", help="""
        The path for the summary, a text file.
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

# string for the final summary file
summary_s = ""

#################################################


    # creating df for player stuff data


#################################################

pitcher_ids = list(df['pitcher'].unique())
pitcher_names = []
pts = ["FF", "SI", "FC", "CH", "FS", "FO", "SC", "CU", "KC", "CS", "SL", "ST", "SV", "KN"]

for id in pitcher_ids:
    pitcher_names.append(df['player_name'][df['pitcher'].to_list().index(id)])

stuff_plus = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'season' : dt.datetime.strptime(df['game_date'][0], "%Y-%m-%d").year,
    'arsenal_avg' : 0,
    'N': 0,
})

location_plus = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'season' : dt.datetime.strptime(df['game_date'][0], "%Y-%m-%d").year,
    'arsenal_avg' : 0,
    'N': 0,
})

pitching_plus = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'season' : dt.datetime.strptime(df['game_date'][0], "%Y-%m-%d").year,
    'arsenal_avg' : 0,
    'N': 0,
})

pitchers = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'pitcher_name' : pitcher_names,
})

for pt in pts:
    # df_pitchers[pt + "_avg_x_rv"] = -1
    stuff_plus[pt + "_avg_x_rv100"] = -1
    pitching_plus[pt + "_avg_x_rv100"] = -1
    location_plus[pt + "_avg_x_rv100"] = -1

    # to calculate arsenal avg later
    stuff_plus[pt + "_n"] = 0
    pitching_plus[pt + "_n"] = 0
    location_plus[pt + "_n"] = 0

#################################################


    # creating new variables


#################################################

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
df['in_play'] = [1 if x == "X" else 0 for x in df['type']]
df['release_pos_x_2'] = [x**2 for x in df['release_pos_x']] 
df['release_pos_z_2'] = [x**2 for x in df['release_pos_z']] 
df['release_pos_y_2'] = [x**2 for x in df['release_pos_y']] 
df['ip_launch_speed'] = [x * y if x == 1 else 0 for x, y in zip(df['in_play'], df['launch_speed'])]
df['ip_launch_angle'] = [x * y if x == 1 else 0 for x, y in zip(df['in_play'], df['launch_angle'])]
df['pitch_number_2'] = [x**2 for x in df['pitch_number']]
df['pfx_x_2'] = [x**2 for x in df['pfx_x']] 
df['pfx_z_2'] = [x**2 for x in df['pfx_z']] 
df['plate_x_2'] = [x**2 for x in df['plate_x']] 
df['plate_z_2'] = [x**2 for x in df['plate_z']] 
df['swing_miss'] = [1 if x == "swinging_strike" else 0 for x in df['description']]
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

        regressors = []

        if tc == "stuff":
            regressors.append(dfs[l]['release_speed'].to_list())
            regressors.append(dfs[l]['release_pos_x'].to_list())
            regressors.append(dfs[l]['release_pos_y'].to_list())
            regressors.append(dfs[l]['release_pos_z'].to_list())
            regressors.append(dfs[l]['pfx_x'].to_list())
            regressors.append(dfs[l]['pfx_z'].to_list())
            regressors.append(dfs[l]['vx0'].to_list())
            regressors.append(dfs[l]['vy0'].to_list())
            regressors.append(dfs[l]['vz0'].to_list())
            regressors.append(dfs[l]['ax'].to_list())
            regressors.append(dfs[l]['ay'].to_list())
            regressors.append(dfs[l]['az'].to_list())
            regressors.append(dfs[l]['release_spin_rate'].to_list())
            regressors.append(dfs[l]['spin_axis'].to_list())
            regressors.append(dfs[l]['release_extension'].to_list())
        elif tc == "location":
            regressors.append(dfs[l]['plate_x'].to_list())
            regressors.append(dfs[l]['plate_z'].to_list())
            regressors.append(dfs[l]['c00'].to_list())
            regressors.append(dfs[l]['c10'].to_list())
            regressors.append(dfs[l]['c20'].to_list())
            regressors.append(dfs[l]['c30'].to_list())
            regressors.append(dfs[l]['c01'].to_list())
            regressors.append(dfs[l]['c11'].to_list())
            regressors.append(dfs[l]['c21'].to_list())
            regressors.append(dfs[l]['c31'].to_list())
            regressors.append(dfs[l]['c02'].to_list())
            regressors.append(dfs[l]['c12'].to_list())
            regressors.append(dfs[l]['c22'].to_list())
            regressors.append(dfs[l]['c32'].to_list())
        elif tc == "pitching":
            regressors.append(dfs[l]['release_speed'].to_list())
            regressors.append(dfs[l]['release_pos_x'].to_list())
            regressors.append(dfs[l]['release_pos_y'].to_list())
            regressors.append(dfs[l]['release_pos_z'].to_list())
            regressors.append(dfs[l]['pfx_x'].to_list())
            regressors.append(dfs[l]['pfx_z'].to_list())
            regressors.append(dfs[l]['vx0'].to_list())
            regressors.append(dfs[l]['vy0'].to_list())
            regressors.append(dfs[l]['vz0'].to_list())
            regressors.append(dfs[l]['ax'].to_list())
            regressors.append(dfs[l]['ay'].to_list())
            regressors.append(dfs[l]['az'].to_list())
            regressors.append(dfs[l]['release_spin_rate'].to_list())
            regressors.append(dfs[l]['spin_axis'].to_list())
            regressors.append(dfs[l]['release_extension'].to_list())
            regressors.append(dfs[l]['plate_x'].to_list())
            regressors.append(dfs[l]['plate_z'].to_list())
            regressors.append(dfs[l]['c00'].to_list())
            regressors.append(dfs[l]['c10'].to_list())
            regressors.append(dfs[l]['c20'].to_list())
            regressors.append(dfs[l]['c30'].to_list())
            regressors.append(dfs[l]['c01'].to_list())
            regressors.append(dfs[l]['c11'].to_list())
            regressors.append(dfs[l]['c21'].to_list())
            regressors.append(dfs[l]['c31'].to_list())
            regressors.append(dfs[l]['c02'].to_list())
            regressors.append(dfs[l]['c12'].to_list())
            regressors.append(dfs[l]['c22'].to_list())
            regressors.append(dfs[l]['c32'].to_list())

        X = []
        y = dfs[l]['delta_run_exp']

        for i in range(len(y)):
            obs = []
            for x in regressors:
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

        X_train = np.array(X)
        y_train = np.array(y)

    #################################################


        # automl


    #################################################


        automl = AutoML()

        automl_settings = {
            "time_budget" : 600,
            "metric" : "r2",
            "task" : "regression",
            "log_file_name" : "ml_stuff.log",
        }

        automl.fit(X_train, y_train, **automl_settings)
        summary_s += str(automl.model.estimator) + "\n"
        summary_s += str(automl.model.estimator.feature_importances_) + "\n"
        summary_s += str(automl.best_config) + "\n"
        summary_s += str(automl.best_loss) + "\n"
        summary_s += str(automl.best_loss_per_estimator) + "\n"
        summary_s += str(automl.metrics_for_best_config) + "\n"

        # expected delta run expectancy
        dfs[l]['x_rv'] = automl.predict(X_train)
        dfs[l]['n_x_rv'] = (dfs[l]['x_rv'] - dfs[l]['x_rv'].mean()) / dfs[l]['x_rv'].std()

    #################################################


        # calculate players' average rvs for each
        # pitch type


    #################################################

        for id in pitcher_ids:
            df_p = dfs[l][dfs[l]['pitcher'] == id]

            for pitch_type in categories_types[l]:
                df_p_p = df_p[df_p['pitch_type'] == pitch_type]

                if len(df_p_p) > 0:
                    avg_x_rv = df_p_p['x_rv'].mean()
                    league_x_rv = dfs[l]['delta_run_exp'].mean()
                    avg_x_rv_v_league = round((avg_x_rv - league_x_rv ) / league_x_rv) + 100

                    # globals mambojambo is means the following
                    # globals()[dataframe name].loc[index, column] = expression
                    globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                            .index, pitch_type + "_avg_x_rv100"] = avg_x_rv_v_league

                    globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                            .index, pitch_type + "_n"] = len(df_p_p)

                    globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                            .index,"N"] += len(df_p_p)

#################################################


    # calculate arsenal averages


#################################################

    for id in pitcher_ids:
        avg_acc = 0

        for pt in pts:
            avg_acc += globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                    .index,pt + '_avg_x_rv100'] \
                    * globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                    .index ,pt + '_n'] \
                    / globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                    .index ,"N"]

        globals()[tc + '_plus'].loc[globals()[tc + '_plus'][globals()[tc + '_plus']['pitcher_id'] == id] \
                .index, 'arsenal_avg'] = round(avg_acc)

    for pt in pts:
        globals()[tc + '_plus'].drop([pt + '_n'], axis=1, inplace=True)

    globals()[tc + '_plus'].drop(['N'], axis=1, inplace=True)

#################################################


    # transfer results to sqlite db
    # new for production and test for dev


#################################################

# conn = sqlite3.connect('new.db')
conn = sqlite3.connect(args.outfile_db)
c = conn.cursor()

c.executescript("""
CREATE TABLE IF NOT EXISTS pitchers(
    pitcher_id INTEGER PRIMARY KEY NOT NULL,
    pitcher_name TEXT
);

CREATE TABLE IF NOT EXISTS stuff_plus(
    stuff_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER NOT NULL,
    FC_avg_x_rv100 INTEGER NOT NULL,
    CH_avg_x_rv100 INTEGER NOT NULL,
    FS_avg_x_rv100 INTEGER NOT NULL,
    FO_avg_x_rv100 INTEGER NOT NULL,
    SC_avg_x_rv100 INTEGER NOT NULL,
    CU_avg_x_rv100 INTEGER NOT NULL,
    KC_avg_x_rv100 INTEGER NOT NULL,
    CS_avg_x_rv100 INTEGER NOT NULL,
    SL_avg_x_rv100 INTEGER NOT NULL,
    ST_avg_x_rv100 INTEGER NOT NULL,
    SV_avg_x_rv100 INTEGER NOT NULL,
    KN_avg_x_rv100 INTEGER NOT NULL,
    arsenal_avg INTEGER NOT NULL,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

CREATE TABLE IF NOT EXISTS location_plus(
    stuff_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER NOT NULL,
    FC_avg_x_rv100 INTEGER NOT NULL,
    CH_avg_x_rv100 INTEGER NOT NULL,
    FS_avg_x_rv100 INTEGER NOT NULL,
    FO_avg_x_rv100 INTEGER NOT NULL,
    SC_avg_x_rv100 INTEGER NOT NULL,
    CU_avg_x_rv100 INTEGER NOT NULL,
    KC_avg_x_rv100 INTEGER NOT NULL,
    CS_avg_x_rv100 INTEGER NOT NULL,
    SL_avg_x_rv100 INTEGER NOT NULL,
    ST_avg_x_rv100 INTEGER NOT NULL,
    SV_avg_x_rv100 INTEGER NOT NULL,
    KN_avg_x_rv100 INTEGER NOT NULL,
    arsenal_avg INTEGER NOT NULL,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

CREATE TABLE IF NOT EXISTS pitching_plus(
    stuff_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER NOT NULL,
    FC_avg_x_rv100 INTEGER NOT NULL,
    CH_avg_x_rv100 INTEGER NOT NULL,
    FS_avg_x_rv100 INTEGER NOT NULL,
    FO_avg_x_rv100 INTEGER NOT NULL,
    SC_avg_x_rv100 INTEGER NOT NULL,
    CU_avg_x_rv100 INTEGER NOT NULL,
    KC_avg_x_rv100 INTEGER NOT NULL,
    CS_avg_x_rv100 INTEGER NOT NULL,
    SL_avg_x_rv100 INTEGER NOT NULL,
    ST_avg_x_rv100 INTEGER NOT NULL,
    SV_avg_x_rv100 INTEGER NOT NULL,
    KN_avg_x_rv100 INTEGER NOT NULL,
    arsenal_avg INTEGER NOT NULL,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

""")

conn.commit()

pitchers.to_sql('pitchers', conn, if_exists='replace', index=False)
location_plus.to_sql('location_plus', conn, if_exists='replace', index=True)
stuff_plus.to_sql('stuff_plus', conn, if_exists='replace', index=True)
pitching_plus.to_sql('pitching_plus', conn, if_exists='replace', index=True)

with open(args.outfile_summary, "w") as text_file:
    text_file.write(summary_s)


