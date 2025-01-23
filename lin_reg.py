import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('data_dirty.csv')

df.dropna()

# creating new variables

# 1 is true
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

c_type = df['type']
c_balls = df['balls']
c_strikes = df['strikes']
c_bb_type = df['bb_type']


# adding run values

run_value = []

for i in range(len(c_type)):
    if c_balls[i] == 3 and c_strikes[i] == 0:
        if c_type[i] == "S":
            run_value.append(-0.078)
        elif c_type[i] == "B":
            run_value.append(0.110)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.314)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.045)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.212)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 3 and c_strikes[i] == 1:
        if c_type[i] == "S":
            run_value.append(-0.083)
        elif c_type[i] == "B":
            run_value.append(0.188)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.197)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.162)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.095)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 2 and c_strikes[i] == 0:
        if c_type[i] == "S":
            run_value.append(-0.067)
        elif c_type[i] == "B":
            run_value.append(0.116)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.171)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.188)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.069)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 3 and c_strikes[i] == 2:
        if c_type[i] == "S":
            run_value.append(-0.349)
        elif c_type[i] == "B":
            run_value.append(0.271)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.131)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.228)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.029)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 1 and c_strikes[i] == 0:
        if c_type[i] == "S":
            run_value.append(-0.053)
        elif c_type[i] == "B":
            run_value.append(0.066)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.109)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.250)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.007)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 2 and c_strikes[i] == 1:
        if c_type[i] == "S":
            run_value.append(-0.076)
        elif c_type[i] == "B":
            run_value.append(0.105)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.107)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.252)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(-0.005)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 0 and c_strikes[i] == 0:
        if c_type[i] == "S":
            run_value.append(-0.044)
        elif c_type[i] == "B":
            run_value.append(0.038)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.074)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.285)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.028)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 1 and c_strikes[i] == 1:
        if c_type[i] == "S":
            run_value.append(-0.067)
        elif c_type[i] == "B":
            run_value.append(0.052)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.061)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.298)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.041)
            else:
                df = df.drop(i)
        else:
            df.drop([i])
    elif c_balls[i] == 2 and c_strikes[i] == 2:
        if c_type[i] == "S":
            run_value.append(-0.251)
        elif c_type[i] == "B":
            run_value.append(0.098)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.046)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.313)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.056)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 0 and c_strikes[i] == 1:
        if c_type[i] == "S":
            run_value.append(-0.062)
        elif c_type[i] == "B":
            run_value.append(0.029)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.038)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.321)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.064)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 1 and c_strikes[i] == 2:
        if c_type[i] == "S":
            run_value.append(-0.208)
        elif c_type[i] == "B":
            run_value.append(0.043)
        elif c_type[i] == "X": 
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.008)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.351)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.094)
            else:
                df.drop([i])
        else:
            df.drop([i])
    elif c_balls[i] == 0 and c_strikes[i] == 2:
        if c_type[i] == "S":
            run_value.append(-0.184)
        elif c_type[i] == "B":
            run_value.append(0.024)
        elif c_type[i] == "X":
            if c_bb_type[i] == "ground_ball":
                run_value.append(-0.013)
            elif c_bb_type[i] == "line_drive":
                run_value.append(0.372)
            elif c_bb_type[i] == "fly_ball" or c_bb_type[i] == "popup":
                run_value.append(0.115)
            else:
                df.drop([i])
        else:
            df.drop([i])
    else:
        df.drop([i])

df['run_value'] = run_value


# get regressors


regressors = []

regressors.append(df['release_speed'].to_list())
regressors.append(df['release_pos_x'].to_list())
regressors.append(df['release_pos_x_2'].to_list())
regressors.append(df['release_pos_y'].to_list())
regressors.append(df['release_pos_y_2'].to_list())
regressors.append(df['release_pos_z'].to_list())
regressors.append(df['release_pos_z_2'].to_list())
regressors.append(df['pfx_x'].to_list())
regressors.append(df['pfx_x_2'].to_list())
regressors.append(df['pfx_z'].to_list())
regressors.append(df['pfx_z_2'].to_list())
regressors.append(df['plate_x'].to_list())
regressors.append(df['plate_x_2'].to_list())
regressors.append(df['plate_z'].to_list())
regressors.append(df['plate_z_2'].to_list())
regressors.append(df['vx0'].to_list())
regressors.append(df['vy0'].to_list())
regressors.append(df['vz0'].to_list())
regressors.append(df['ax'].to_list())
regressors.append(df['ay'].to_list())
regressors.append(df['az'].to_list())
regressors.append(df['release_spin_rate'].to_list())
regressors.append(df['spin_axis'].to_list())
regressors.append(df['release_extension'].to_list())
regressors.append(df['in_play'].to_list())
regressors.append(df['ip_launch_speed'].to_list())
regressors.append(df['ip_launch_angle'].to_list())
regressors.append(df['pitch_number'].to_list())
regressors.append(df['pitch_number_2'].to_list())
regressors.append(df['balls'].to_list())
regressors.append(df['strikes'].to_list())
regressors.append(df['effective_speed'].to_list())

X = []
y = run_value
to_drop = []


# drop observations with nans


for i in range(len(y)):
    obs = []
    for x in regressors:
        obs.append(x[i])
    X.append(obs)

for i in np.argwhere(np.isnan(np.array(X))):
    if i[0]+1 not in to_drop:
        to_drop.append(i[0]+1)

to_drop = list(reversed(to_drop))

for i in to_drop:
    X.pop(i-1)
    y.pop(i-1)
    df = df.drop([i])

print("Observations dropped due to missing values:", len(to_drop))


pts = ["FF", "SI", "FC", "CH", "FS", "FO", "SC", "CU", "KC", "CS", "SL", "ST", "SV", "KN"]

pitch_type = df['pitch_type'].to_list()

for pt in pts:
    print(pt)
    ind = []
    dep = []

    for i in range(len(pitch_type)):
        if pitch_type[i] == pt:
            ind.append(X[i])
            dep.append(y[i])

    if len(ind) < 1:
        print("Not enough observations to calculate:", pt, ". Observations:", len(ind))
    else:
        regr = linear_model.LinearRegression()
        regr.fit(ind , dep)






