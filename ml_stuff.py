import numpy as np
import pandas as pd
import sqlite3
from flaml import AutoML

df = pd.read_csv('data_dirty.csv')

df.dropna()

#################################################


    # creating df for player stuff data


#################################################

pitcher_ids = list(df['pitcher'].unique())
pitcher_names = []
pts = ["FF", "SI", "FC", "CH", "FS", "FO", "SC", "CU", "KC", "CS", "SL", "ST", "SV", "KN"]

for id in pitcher_ids:
    pitcher_names.append(df['player_name'][df['pitcher'].to_list().index(id)])


df_pitchers = pd.DataFrame({
    'pitcher_id' : pitcher_ids,
    'pitcher_name' : pitcher_names,
})

for pt in pts:
    # df_pitchers[pt + "_avg_x_rv"] = -1
    df_pitchers[pt + "_avg_x_rv100"] = -1

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

#################################################


    # separate into three different categories


#################################################

df_f = df[df['category'] == "fastball"]
df_o = df[df['category'] == "offspeed"]
df_b = df[df['category'] == "breaking_ball"]

df_f = df_f.reset_index(drop=True)
df_o = df_o.reset_index(drop=True)
df_b = df_b.reset_index(drop=True)

dfs = [df_f, df_o, df_b]

#################################################


    # get regressors


#################################################

p_l_avg = []

for l in range(len(dfs)):

    regressors = []

    regressors.append(dfs[l]['release_speed'].to_list())
    regressors.append(dfs[l]['release_pos_x'].to_list())
    regressors.append(dfs[l]['release_pos_x_2'].to_list())
    regressors.append(dfs[l]['release_pos_y'].to_list())
    regressors.append(dfs[l]['release_pos_y_2'].to_list())
    regressors.append(dfs[l]['release_pos_z'].to_list())
    regressors.append(dfs[l]['release_pos_z_2'].to_list())
    regressors.append(dfs[l]['pfx_x'].to_list())
    regressors.append(dfs[l]['pfx_x_2'].to_list())
    regressors.append(dfs[l]['pfx_z'].to_list())
    regressors.append(dfs[l]['pfx_z_2'].to_list())
    regressors.append(dfs[l]['vx0'].to_list())
    regressors.append(dfs[l]['vy0'].to_list())
    regressors.append(dfs[l]['vz0'].to_list())
    regressors.append(dfs[l]['ax'].to_list())
    regressors.append(dfs[l]['ay'].to_list())
    regressors.append(dfs[l]['az'].to_list())
    regressors.append(dfs[l]['release_spin_rate'].to_list())
    regressors.append(dfs[l]['spin_axis'].to_list())
    regressors.append(dfs[l]['release_extension'].to_list())

    X = []
    y = dfs[l]['delta_run_exp'].to_list()

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
        if i[0]+1 not in to_drop:
            to_drop.append(i[0]+1)

    for i in np.argwhere(np.isnan(np.array(X))):
        if i[0]+1 not in to_drop:
            to_drop.append(i[0]+1)

    to_drop = list(reversed(sorted(to_drop)))

    for i in to_drop:
        X.pop(i-1)
        y.pop(i-1)
        category.pop(i-1)
        dfs[l] = dfs[l].drop([i])

    print("Observations dropped due to missing values:", len(to_drop))

    X_train = np.array(X)
    y_train = np.array(y)

#################################################


    # automl


#################################################

    automl = AutoML()

    automl_settings = {
        "time_budget" : 7200,
        "metric" : "r2",
        "task" : "regression",
        "log_file_name" : "ml_stuff.log",
    }

    automl.fit(X_train, y_train, **automl_settings)
    print(automl.model.estimator)

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
                p_l_avg.append(league_x_rv)

                # df_pitchers[pitch_type + "_avg_x_rv"][df_pitchers[df_pitchers['pitcher_id'] == id].index] = avg_x_rv
                df_pitchers[pitch_type + "_avg_x_rv100"][df_pitchers[df_pitchers['pitcher_id'] == id].index] = round(((avg_x_rv - league_x_rv ) / league_x_rv)+ 100)

#################################################


    # transfer results to sqlite db


#################################################

conn = sqlite3.connect('new.db')
c = conn.cursor()

df_pitchers.to_sql('pitchers', conn, if_exists='replace', index=False)



