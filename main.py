import pandas as pd

# Model training and prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB


# Create a function to train and predict for each season and print out the MSE based on Random Forest Regressor
def train_predict_RFR(df_train, df_test):
    x_train = df_train[['lat', 'lng']]
    y_train = df_train['aqi']
    x_test = df_test[['lat', 'lng']]
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    model.fit(x_train, y_train)
    # Predict aqi integers to the test data
    y_pred = model.predict(x_test).astype(int)
    print("MSE: ", mean_squared_error(y_train, model.predict(x_train)))
    return y_pred


# function to train and predict for each season and print out the MSE based on Naive Bayes
def train_predict_NB(df_train, df_test):
    x_train = df_train[['lat', 'lng']]
    y_train = df_train['aqi']
    x_test = df_test[['lat', 'lng']]
    model = GaussianNB()
    model.fit(x_train, y_train)
    # Predict aqi integers to the test data
    y_pred = model.predict(x_test).astype(int)
    print("MSE: ", mean_squared_error(y_train, model.predict(x_train)))
    return y_pred


if __name__ == '__main__':
    filepath_train = r"Train.csv"
    filepath_test = r"Test.csv"

    # Read train data from CVS file with only the columns of ['date', 'lat', 'long', 'aqi']
    df = pd.read_csv(filepath_train, usecols=['date', 'lat', 'lng', 'aqi'])

    # divide date to 4 seasons, 1 for spring (January-March), 2 for summer (April-June), 3 for fall (July-September),
    # 4 for winter (October-December)
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['date'].dt.month.apply(
        lambda x: 1 if x in [1, 2, 3] else 2 if x in [4, 5, 6] else 3 if x in [7, 8, 9] else 4)

    # get 4 separate dataframes for each season
    df_spring = df[df['season'] == 1]
    df_summer = df[df['season'] == 2]
    df_fall = df[df['season'] == 3]
    df_winter = df[df['season'] == 4]

    # Read test data from CVS file with only the columns of ['season', 'lat', 'lng']
    df_test = pd.read_csv(filepath_test, usecols=['season', 'lat', 'lng', 'ID'])

    # get 4 separate dataframes for each season
    df_test_spring = df_test[df_test['season'] == 1]
    df_test_summer = df_test[df_test['season'] == 2]
    df_test_fall = df_test[df_test['season'] == 3]
    df_test_winter = df_test[df_test['season'] == 4]

    # Predict for each season based on Naive Bayes
    y_pred_spring = train_predict_NB(df_spring, df_test_spring)
    y_pred_summer = train_predict_NB(df_summer, df_test_summer)
    y_pred_fall = train_predict_NB(df_fall, df_test_fall)
    y_pred_winter = train_predict_NB(df_winter, df_test_winter)

    # Combine the prediction results for each season into a matrix of 4 columns
    y_pred = pd.DataFrame(
        {'spring': y_pred_spring, 'summer': y_pred_summer, 'fall': y_pred_fall, 'winter': y_pred_winter})
    # transpose the matrix and merge to a vector
    y_pred = y_pred.T
    y_pred = y_pred.unstack().to_frame()
    # add column headers as "aqi"
    y_pred.columns = ['aqi']
    # remove index and keep only the aqi column
    y_pred = y_pred.reset_index(drop=True)

    # Add the ID column to the dataframe with the same order as in test.csv file
    y_pred.insert(0, 'ID', df_test['ID'])

    # Save the prediction results to a CSV file
    y_pred.to_csv('submission.csv', index=False)
