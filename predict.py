import numpy as np
import datetime as dt
import pandas as pd
import sqlite3
import sys
import argparse
import pickle

#################################################


    # parsing flags and opening csv


#################################################

parser = argparse.ArgumentParser(description="""
        usage: python predict.py [-h] [-odb data.db] [-olog sum.log]
               [-models ./path-to models-folder] infile.csv 

        Script to calculate stuff, location and pitching plus from
        baseball savant data and models created with ml_stuff. The
        input data should be in csv format and the path to the
        folder containing the models. The output is a sqlite db
        containing  the resulting stats for each pitcher included
        in the original csv.

""")

parser.add_argument("input_file_path")
parser.add_argument("-odb", "--outfile-db", dest="outfile_db", default="predict.db", help="""
        The path for the outfile, the sqlite db with the results.
""")
parser.add_argument("-ol", "--outfile-log", dest="outfile_summary", default="predict.log", help="""
        The path for the log, a text file.
""")
parser.add_argument("-m", "--models", dest="models_path", default="./models", help="""
        The path for models folder, default is ./models .
""")

args = parser.parse_args()

if len(sys.argv) < 2:
    print("""
    There was no path provided for the baseball savant data.
    Run 
          python predict.py path_to_bs_data.csv
    or
          python predict.py --help
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

# string for the final summary file
summary_s = ""

fastballs = ["FF", "SI"]
offspeeds = ["FC", "CH", "FS", "FO", "SC"]
breaking_balls = ["CU", "KC", "CS", "SL", "ST", "SV", "KN"]
categories = ["fastball", "offspeed", "breaking_ball"]
categories_types = [fastballs, offspeeds, breaking_balls]

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


season = []
for d in df['game_date']:
    season.append(dt.datetime.strptime(d, "%Y-%m-%d").year)

df['season'] = season
seasons = list(df['season'].unique())
pitcher_ids = list(df['pitcher'].unique())
pitcher_names = []
pitcher_handedness = []
pts = ["FF", "SI", "FC", "CH", "FS", "FO", "SC", "CU", "KC", "CS", "SL", "ST", "SV", "KN"]


stuff_plus = pd.DataFrame({
    'season': pd.Series(dtype = "int"),
    'FF_avg_x_rv100': pd.Series(dtype = "int"),
    'SI_avg_x_rv100': pd.Series(dtype = "int"),
    'FC_avg_x_rv100': pd.Series(dtype = "int"),
    'CH_avg_x_rv100': pd.Series(dtype = "int"),
    'FS_avg_x_rv100': pd.Series(dtype = "int"),
    'FO_avg_x_rv100': pd.Series(dtype = "int"),
    'SC_avg_x_rv100': pd.Series(dtype = "int"),
    'CU_avg_x_rv100': pd.Series(dtype = "int"),
    'KC_avg_x_rv100': pd.Series(dtype = "int"),
    'CS_avg_x_rv100': pd.Series(dtype = "int"),
    'SL_avg_x_rv100': pd.Series(dtype = "int"),
    'ST_avg_x_rv100': pd.Series(dtype = "int"),
    'SV_avg_x_rv100': pd.Series(dtype = "int"),
    'KN_avg_x_rv100': pd.Series(dtype = "int"),
    'FF_n': pd.Series(dtype = "int"),
    'SI_n': pd.Series(dtype = "int"),
    'FC_n': pd.Series(dtype = "int"),
    'CH_n': pd.Series(dtype = "int"),
    'FS_n': pd.Series(dtype = "int"),
    'FO_n': pd.Series(dtype = "int"),
    'SC_n': pd.Series(dtype = "int"),
    'CU_n': pd.Series(dtype = "int"),
    'KC_n': pd.Series(dtype = "int"),
    'CS_n': pd.Series(dtype = "int"),
    'SL_n': pd.Series(dtype = "int"),
    'ST_n': pd.Series(dtype = "int"),
    'SV_n': pd.Series(dtype = "int"),
    'KN_n': pd.Series(dtype = "int"),
    'N': pd.Series(dtype = "int"),
    'arsenal_avg': pd.Series(dtype = "int"),
    'pitcher_id': pd.Series(dtype = "int"),
})

location_plus = pd.DataFrame({
    'season': pd.Series(dtype = "int"),
    'FF_avg_x_rv100': pd.Series(dtype = "int"),
    'SI_avg_x_rv100': pd.Series(dtype = "int"),
    'FC_avg_x_rv100': pd.Series(dtype = "int"),
    'CH_avg_x_rv100': pd.Series(dtype = "int"),
    'FS_avg_x_rv100': pd.Series(dtype = "int"),
    'FO_avg_x_rv100': pd.Series(dtype = "int"),
    'SC_avg_x_rv100': pd.Series(dtype = "int"),
    'CU_avg_x_rv100': pd.Series(dtype = "int"),
    'KC_avg_x_rv100': pd.Series(dtype = "int"),
    'CS_avg_x_rv100': pd.Series(dtype = "int"),
    'SL_avg_x_rv100': pd.Series(dtype = "int"),
    'ST_avg_x_rv100': pd.Series(dtype = "int"),
    'SV_avg_x_rv100': pd.Series(dtype = "int"),
    'KN_avg_x_rv100': pd.Series(dtype = "int"),
    'FF_n': pd.Series(dtype = "int"),
    'SI_n': pd.Series(dtype = "int"),
    'FC_n': pd.Series(dtype = "int"),
    'CH_n': pd.Series(dtype = "int"),
    'FS_n': pd.Series(dtype = "int"),
    'FO_n': pd.Series(dtype = "int"),
    'SC_n': pd.Series(dtype = "int"),
    'CU_n': pd.Series(dtype = "int"),
    'KC_n': pd.Series(dtype = "int"),
    'CS_n': pd.Series(dtype = "int"),
    'SL_n': pd.Series(dtype = "int"),
    'ST_n': pd.Series(dtype = "int"),
    'SV_n': pd.Series(dtype = "int"),
    'KN_n': pd.Series(dtype = "int"),
    'N': pd.Series(dtype = "int"),
    'arsenal_avg': pd.Series(dtype = "int"),
    'pitcher_id': pd.Series(dtype = "int"),
})

pitching_plus = pd.DataFrame({
    'season': pd.Series(dtype = "int"),
    'FF_avg_x_rv100': pd.Series(dtype = "int"),
    'SI_avg_x_rv100': pd.Series(dtype = "int"),
    'FC_avg_x_rv100': pd.Series(dtype = "int"),
    'CH_avg_x_rv100': pd.Series(dtype = "int"),
    'FS_avg_x_rv100': pd.Series(dtype = "int"),
    'FO_avg_x_rv100': pd.Series(dtype = "int"),
    'SC_avg_x_rv100': pd.Series(dtype = "int"),
    'CU_avg_x_rv100': pd.Series(dtype = "int"),
    'KC_avg_x_rv100': pd.Series(dtype = "int"),
    'CS_avg_x_rv100': pd.Series(dtype = "int"),
    'SL_avg_x_rv100': pd.Series(dtype = "int"),
    'ST_avg_x_rv100': pd.Series(dtype = "int"),
    'SV_avg_x_rv100': pd.Series(dtype = "int"),
    'KN_avg_x_rv100': pd.Series(dtype = "int"),
    'FF_n': pd.Series(dtype = "int"),
    'SI_n': pd.Series(dtype = "int"),
    'FC_n': pd.Series(dtype = "int"),
    'CH_n': pd.Series(dtype = "int"),
    'FS_n': pd.Series(dtype = "int"),
    'FO_n': pd.Series(dtype = "int"),
    'SC_n': pd.Series(dtype = "int"),
    'CU_n': pd.Series(dtype = "int"),
    'KC_n': pd.Series(dtype = "int"),
    'CS_n': pd.Series(dtype = "int"),
    'SL_n': pd.Series(dtype = "int"),
    'ST_n': pd.Series(dtype = "int"),
    'SV_n': pd.Series(dtype = "int"),
    'KN_n': pd.Series(dtype = "int"),
    'N': pd.Series(dtype = "int"),
    'arsenal_avg': pd.Series(dtype = "int"),
    'pitcher_id': pd.Series(dtype = "int"),
})

for id in pitcher_ids:
    pitcher_names.append(df['player_name'][df['pitcher'].to_list().index(id)])
    pitcher_handedness.append(df['p_throws'][df['pitcher'].to_list().index(id)])

pitchers = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'pitcher_name' : pitcher_names,
    'p_throws' : pitcher_handedness,
})

stuff_regressors = pd.DataFrame({
    'pitcher_id' : pd.Series(dtype = "int"),
    'pitch_type' : pd.Series(dtype = "str"),
    'season' : pd.Series(dtype = "int"),
    'release_speed': pd.Series(dtype = "float"),
    'release_pos_x': pd.Series(dtype = "float"),
    'release_pos_y': pd.Series(dtype = "float"),
    'release_pos_z': pd.Series(dtype = "float"),
    'pfx_x': pd.Series(dtype = "float"),
    'pfx_z': pd.Series(dtype = "float"),
    'vx0': pd.Series(dtype = "float"),
    'vy0': pd.Series(dtype = "float"),
    'vz0': pd.Series(dtype = "float"),
    'ax': pd.Series(dtype = "float"),
    'ay': pd.Series(dtype = "float"),
    'az': pd.Series(dtype = "float"),
    'release_spin_rate': pd.Series(dtype = "float"),
    'spin_axis': pd.Series(dtype = "float"),
    'release_extension': pd.Series(dtype = "float"),
    'stuff_plus': pd.Series(dtype = "int"),
})

#################################################


    # save average for each regressor for each
    # pitcher


#################################################

for id in pitcher_ids:
    df_pitcher = df[df['pitcher'] == id]

    for s in seasons:
        df_p_s = df_pitcher[df_pitcher['season'] == s]

        if len(df_p_s) > 0:
            for pt in pts:
                df_pitcher_pt = df_p_s[df_p_s['pitch_type'] == pt]

                if len(df_pitcher_pt) > 0:
                    stuff_regressors = pd.concat([pd.DataFrame([[
                        int(id), 
                        pt,
                        s, 
                        round(df_pitcher_pt['release_speed'].mean(), 2),
                        round(df_pitcher_pt['release_pos_x'].mean(), 2),
                        round(df_pitcher_pt['release_pos_y'].mean(), 2),
                        round(df_pitcher_pt['release_pos_z'].mean(), 2),
                        round(df_pitcher_pt['pfx_x'].mean(), 2),
                        round(df_pitcher_pt['pfx_z'].mean(), 2),
                        round(df_pitcher_pt['vx0'].mean(), 2),
                        round(df_pitcher_pt['vy0'].mean(), 2),
                        round(df_pitcher_pt['vz0'].mean(), 2),
                        round(df_pitcher_pt['ax'].mean(), 2),
                        round(df_pitcher_pt['ay'].mean(), 2),
                        round(df_pitcher_pt['az'].mean(), 2),
                        round(df_pitcher_pt['release_spin_rate'].mean(), 2),
                        round(df_pitcher_pt['spin_axis'].mean(), 2),
                        round(df_pitcher_pt['release_extension'].mean(), 2),
                        -1,
                    ]], columns = stuff_regressors.columns), stuff_regressors], ignore_index=True)

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

    #################################################


        # predict


    #################################################

        with open(f"{args.models_path}/{tc}_{categories[l]}.pkl", 'rb') as f:
            loaded_automl = pickle.load(f)

        dfs[l]['x_rv'] = loaded_automl.predict(X)

    #################################################


        # calculate players' average rvs for each
        # pitch type


    #################################################

for tc in to_calculate:
    dfs = df_groups

    for s in seasons:
        for id in pitcher_ids:
            to_append = [
                s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, id,
            ]

            for l in range(len(dfs)):
                df_p = dfs[l][dfs[l]['pitcher'] == id]

                if len(df_p) > 0:
                    for pitch_type in categories_types[l]:
                        df_p_p = df_p[df_p['pitch_type'] == pitch_type]

                        if len(df_p_p) > 0:
                            ind_x = list(globals()[tc + '_plus'].columns).index(pitch_type + '_avg_x_rv100')
                            ind_n = list(globals()[tc + '_plus'].columns).index(pitch_type + '_n')
                            ind_N = list(globals()[tc + '_plus'].columns).index('N')

                            avg_x_rv = df_p_p['x_rv'].mean()
                            league_x_rv = dfs[l]['x_rv'].mean()
                            avg_x_rv_v_league = round(((avg_x_rv - league_x_rv ) / league_x_rv) + 100)

                            to_append[ind_x] = avg_x_rv_v_league
                            to_append[ind_n] = len(df_p_p)
                            to_append[ind_N] += len(df_p_p)

                            stuff_regressors.loc[stuff_regressors[(stuff_regressors['pitch_type'] == pitch_type) & \
                                    (stuff_regressors['pitcher_id'] == id) & (stuff_regressors['season'] == s)].index, 
                                                 'stuff_plus'] = avg_x_rv_v_league

                    globals()[tc + '_plus'] = pd.concat([pd.DataFrame([to_append], columns = globals()[tc + '_plus'].columns), 
                                                         globals()[tc + '_plus']], ignore_index=True)

#################################################


    # calculate arsenal averages


#################################################

for tc in to_calculate:
    avg_ars = globals()[tc + '_plus']['arsenal_avg']

    for pt in pts:
        avg_ars += pd.Series([round((x * y) / z) for x, y, z in \
                zip(globals()[tc + '_plus'][pt + '_avg_x_rv100'], globals()[tc + '_plus'][pt + '_n'], globals()[tc + '_plus']['N'])])

    globals()[tc + '_plus']['arsenal_avg'] = avg_ars

#################################################


    # transfer results to sqlite db
    # new for production and test for dev


#################################################

# conn = sqlite3.connect('new.db')
conn = sqlite3.connect(args.outfile_db)
c = conn.cursor()

c.executescript("""
CREATE TABLE IF NOT EXISTS pitchers(
    pitcher_id INTEGER PRIMARY KEY,
    pitcher_name TEXT,
    p_throws TEXT
);

CREATE TABLE IF NOT EXISTS stuff_regressors(
    regressor_id INTEGER PRIMARY KEY,
    pitcher_id INTEGER NOT NULL,
    pitch_type INTEGER NOT NULL,
    season INTEGER NOT NULL,
    release_speed INTEGER,
    release_pos_x INTEGER,
    release_pos_y INTEGER,
    release_pos_z INTEGER,
    pfx_x INTEGER,
    pfx_z INTEGER,
    vx0 INTEGER,
    vy0 INTEGER,
    vz0 INTEGER,
    ax INTEGER,
    ay INTEGER,
    az INTEGER,
    release_spin_rate INTEGER,
    spin_axis INTEGER,
    release_extension INTEGER,
    stuff_plus INTEGER,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

CREATE TABLE IF NOT EXISTS stuff_plus(
    stuff_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER,
    SI_avg_x_rv100 INTEGER,
    FC_avg_x_rv100 INTEGER,
    CH_avg_x_rv100 INTEGER,
    FS_avg_x_rv100 INTEGER,
    FO_avg_x_rv100 INTEGER,
    SC_avg_x_rv100 INTEGER,
    CU_avg_x_rv100 INTEGER,
    KC_avg_x_rv100 INTEGER,
    CS_avg_x_rv100 INTEGER,
    SL_avg_x_rv100 INTEGER,
    ST_avg_x_rv100 INTEGER,
    SV_avg_x_rv100 INTEGER,
    KN_avg_x_rv100 INTEGER,
    FF_n INTEGER,
    SI_n INTEGER,
    FC_n INTEGER,
    CH_n INTEGER,
    FS_n INTEGER,
    FO_n INTEGER,
    SC_n INTEGER,
    CU_n INTEGER,
    KC_n INTEGER,
    CS_n INTEGER,
    SL_n INTEGER,
    ST_n INTEGER,
    SV_n INTEGER,
    KN_n INTEGER,
    N INTEGER,
    arsenal_avg INTEGER,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

CREATE TABLE IF NOT EXISTS location_plus(
    location_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER,
    SI_avg_x_rv100 INTEGER,
    FC_avg_x_rv100 INTEGER,
    CH_avg_x_rv100 INTEGER,
    FS_avg_x_rv100 INTEGER,
    FO_avg_x_rv100 INTEGER,
    SC_avg_x_rv100 INTEGER,
    CU_avg_x_rv100 INTEGER,
    KC_avg_x_rv100 INTEGER,
    CS_avg_x_rv100 INTEGER,
    SL_avg_x_rv100 INTEGER,
    ST_avg_x_rv100 INTEGER,
    SV_avg_x_rv100 INTEGER,
    KN_avg_x_rv100 INTEGER,
    FF_n INTEGER,
    SI_n INTEGER,
    FC_n INTEGER,
    CH_n INTEGER,
    FS_n INTEGER,
    FO_n INTEGER,
    SC_n INTEGER,
    CU_n INTEGER,
    KC_n INTEGER,
    CS_n INTEGER,
    SL_n INTEGER,
    ST_n INTEGER,
    SV_n INTEGER,
    KN_n INTEGER,
    N INTEGER,
    arsenal_avg INTEGER,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

CREATE TABLE IF NOT EXISTS pitching_plus(
    pitching_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    FF_avg_x_rv100 INTEGER,
    SI_avg_x_rv100 INTEGER,
    FC_avg_x_rv100 INTEGER,
    CH_avg_x_rv100 INTEGER,
    FS_avg_x_rv100 INTEGER,
    FO_avg_x_rv100 INTEGER,
    SC_avg_x_rv100 INTEGER,
    CU_avg_x_rv100 INTEGER,
    KC_avg_x_rv100 INTEGER,
    CS_avg_x_rv100 INTEGER,
    SL_avg_x_rv100 INTEGER,
    ST_avg_x_rv100 INTEGER,
    SV_avg_x_rv100 INTEGER,
    KN_avg_x_rv100 INTEGER,
    FF_n INTEGER,
    SI_n INTEGER,
    FC_n INTEGER,
    CH_n INTEGER,
    FS_n INTEGER,
    FO_n INTEGER,
    SC_n INTEGER,
    CU_n INTEGER,
    KC_n INTEGER,
    CS_n INTEGER,
    SL_n INTEGER,
    ST_n INTEGER,
    SV_n INTEGER,
    KN_n INTEGER,
    N INTEGER,
    arsenal_avg INTEGER,
    pitcher_id INTEGER NOT NULL,
    FOREIGN KEY(pitcher_id) REFERENCES pitchers(pitcher_id)
);

""")

conn.commit()

pitchers.to_sql('pitchers', conn, if_exists='append', index=False)
stuff_regressors.to_sql('stuff_regressors', conn, if_exists='append', index=True, index_label='regressor_id')
location_plus.to_sql('location_plus', conn, if_exists='append', index=True, index_label='location_id')
stuff_plus.to_sql('stuff_plus', conn, if_exists='append', index=True, index_label='stuff_id')
pitching_plus.to_sql('pitching_plus', conn, if_exists='append', index=True, index_label='pitching_id')

with open(args.outfile_summary, "w") as text_file:
    text_file.write(summary_s)
