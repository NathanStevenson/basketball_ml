# Nathan Stevenson NBA Win Predictions with Machine Learning followed tutorial @: https://www.youtube.com/watch?v=egTylm6C2is

import pandas as pd

# ------- PROCESSING/CLEANING THE DATA FOR MACHINE LEARNING w/ PANDAS 

# This reads a csv file that is stored in the same directory as the jupyter notebook
df = pd.read_csv("nba_games.csv", index_col=0)
# this sorts the values by date and then deletes the old index and reassigns it a new one
df = df.sort_values("date")
df = df.reset_index(drop=True)

# you can remove extra columns that are needed with these cmds
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

# function that adds a target. The target is whether the team won or loss the next game (goal is to use columns to predict next game)
def add_target(team):
  # team["target"] adds a column, assigning it the value if the shift is negative it is pulling it from future games
  team["target"] = team["won"].shift(-1)
  return team

# we want to split it up by team so that when we call the function it is that teams next game and not some other random team
df = df.groupby("team", group_keys=False).apply(add_target)

# df["team"] (this outputs only the values in the column "team")
# allows us to filter our dataframe with only teams named Washington

# this is going to find all of the null values in the df. (last one doesnt have a next game want to parse it) assign it (2)
df["target"][pd.isnull(df["target"])] = 2

# we then want all the falses to be 0 and the trues to be 1 so we are going to convert the type from bool to int
df["target"] = df["target"].astype(int, errors="ignore")

# column.value_counts() this outputs the number of each type in the specified columns

# find all of the isnull values they will be set to True rather than False
nulls = pd.isnull(df)
# sum all of the null values so that you can see which columns have null values 
nulls = nulls.sum()
# this will then filter all columns that have some null values (if some columns have many then just remove the column)
nulls = nulls[nulls > 0]

# only keep the valid cols if the column is not in the nulls list
valid_cols = df.columns[~df.columns.isin(nulls.index)]

# whenever we assign the df back to itself it is good practice to give it the copy so you dont run into issues
df = df[valid_cols].copy()

# ------------- MACHINE LEARNING PORTION 
# this the the way we are oging to perform feature selection
# another method is called cross validation where you split the data up and train the model on a portion of your data and then test it on the other
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
# using ridge regression to perform the feature selection (good in cases of high multicollinearity)
from sklearn.linear_model import RidgeClassifier

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

# selector is going to train the model by using different columns and seeing how well the model does with those columns
# standard practice to pick about 25% of features to select but this number can be played around with
# forward starts with 0 features and continuously adds the new feature that improves the model the best
# cv is used for cross validation which is the TimeSeries split we defined above
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)

# all of the columns we are not going to scale (they are strings or metadata cols)
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_cols = df.columns[~df.columns.isin(removed_columns)]

from sklearn.preprocessing import MinMaxScaler
# this is the way to scale all of the values to be between 0 and 1 which works best for Ridge Regression
scaler = MinMaxScaler()
df[selected_cols] = scaler.fit_transform(df[selected_cols])

# we are fitting the model with the 30 best parameters at predicting the target (takes a while)
sfs.fit(df[selected_cols], df["target"])

# if feature selector has picked this column in the model will evaluate to true so we can use this to get the cols
predictors = list(selected_cols[sfs.get_support()])

# time series data use past data to predict new data
# step=1 means we are increasing the season one at a time (need at least 2 seasons of training to predict 1 season)
def backtest(data, model, predictors, start=2, step=1):
  all_predictions = []
  seasons = sorted(data["season"].unique())
  for i in range(start, len(seasons), step):
    season = seasons[i]
    # we are going to train our model with all of the seasons that come before the current season
    train = data[data["season"] < season]
    # test it on the current season to see how it performs
    test = data[data["season"] == season]

    # fit the model with the predictors and then have it determine which of these correlate to the target
    model.fit(train[predictors], train["target"])

    # this is going to make predictions on the next season using the model
    preds = model.predict(test[predictors])
    # by default model.predict returns a numpy array that we are going to transform into a panda series
    preds = pd.Series(preds, index=test.index)
    # this is going to combine two dataframes and they should be treated as separate cols as specified by axis=1
    combined = pd.concat([test["target"], preds], axis=1)
    # renaming the cols
    combined.columns = ["actual", "prediction"]
    all_predictions.append(combined)

    return pd.concat(all_predictions)

predictions = backtest(df, rr, predictors)


from sklearn.metrics import accuracy_score

accuracy_score(predictions["actual"], predictions["prediction"])

# want to set a baseline for what is a good accuracy for predicting wins
# this is going to group all the times the team is at home and they won and then divide it by the total number
# home team wins 57% of the time so we want to be better than at least 57%
df.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

from math import isnan
# we want to have rolling averages. see how a team has performed over the course of their last 10 games
df_rolling = df[list(selected_cols) + ["won", "team", "season"]]

def find_team_averages(team):
  # .rolling is a pandas method that groups the previous 10 values to make the prediction and averages them
  rolling = team.rolling(10).mean()
  return rolling

# want to only consider the performance of a team and season to make sure it is the same team youre comparing
# you can use .apply() function on a dataframe to apply the function to all the rows (the first 10 values are null b/c the previous 10 dont exist)
df_rolling = df.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

# this is going to take all the columns and append _10 to their name
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
# this is going to combine our normaly dataframes with the new rolling average data frames (axis=1 uses the columns to combine axis=0 uses rows)
df = pd.concat([df, df_rolling], axis=1)

# drops any rows with missing cols
df = df.dropna()

# this function allows the user to specify which function they want to shift and then grabs that teams value and gets the next game
def shift_col(team, col_name):
  # takes value from the next game and shifts it back one row
  next_col = team[col_name].shift(-1)
  return next_col

def add_col(df, col_name):
  # lambda (x) x becomes the value that is essentially the iterator on the loop. since we are looping over teams x will be the specific team
  return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

# create a column called "home_next" which will add whether their next game is home or away so the algorithm knows
# be careful only doing this for data that you will know for the next game
df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# pull in information about how the opponent has performed over the past 10 games
# merge the next teams info who a team is playing into the same row as the current team

# this will remove rows where it cannot find the opponents next games. (to predict future games we have to replace these w/ values of who they play next)
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                left_on=["team", "date_next"], 
                right_on=["team_opp_next", "date_next"])
# _y is the data from the opposing teams and _x or nothing is the original team

# this pulls out all non numeric types (what we dont wanna pass into the model)
# outside of non-numeric data we also want to remove any target data as that is what we are trying to predict and in the real world we will not have it
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns + ['target_10_x', 'target_10_y']
selected_cols = full.columns[~full.columns.isin(removed_columns)]

# this will spit out the 30 best features that are best for prediciting the target
# any time you fit the model it is going to take a long time to run
sfs.fit(full[selected_cols], full["target"])

# if feature selector has picked this column in the model will evaluate to true so we can use this to get the cols
predictors = list(selected_cols[sfs.get_support()])

predictions = backtest(full, rr, predictors)
accuracy_score(predictions["actual"], predictions["prediction"])

# as of rn this model predicts the winner of the next game with 64.9% would like to get this into the mid/high 70s range

# GOING FORWARD 
# all of the code above was used explicitly from the provided tutorial but the code that comes next I plan to write by myself in order to improve the model and predict future games
# PLANS:
# Improve the accuracy by try using other models (non-ridge: HexGBoost), try different number of features and backward selections, look at different amounts of last n games
# Predict future rows by providing the missing values (web-scrape the new up to date scores and fill in the upcoming schedules for the teams)

# This model predicts the likelihood that a team wins or loses a game (binary) I am going to watch another ML tutorial to predict a continuous variable
# namely the price of housing. Then I hope to take that knowledge combined w/ the knowledge from this tutorial to code more specific predictions such as the 
# points scored and maybe even the number of points scored per player and start to dive into some sports betting categories