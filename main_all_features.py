import pandas as pd
import matplotlib.pyplot as plt

filepath_train = r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\Train.csv"
filepath_test = r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\Test.csv"

attributes_train = [
    "date", "lat", "lng", "temperature", "precipitation", "humidity",
    "global_radiation", "hydrometric_level", "N", "NE", "E", "SE", "S", "SW",
    "W", "NW", "type", "pm25", "pm10", "o3", "so2", "no2", "pm25_aqi",
    "pm10_aqi", "no2_aqi", "o3_aqi", "so2_aqi", "aqi", "utm_x", "utm_y",
    "dtm_milan", "aspect", "dusaf15", "geologia", "hillshade", "ndvi_2019",
    "plan_curvature", "profile_curvature", "water_distance", "slope", "spi",
    "tri", "twi", "geo_0", "geo_1", "geo_2", "geo_3", "geo_4", "geo_5", "geo_6",
    "lc_11", "lc_12", "lc_14", "lc_21", "lc_22", "lc_23", "lc_31", "lc_32",
    "lc_33", "lc_41", "lc_51"
]

# Read train data from CVS file with only the columns of ['date', 'lat', 'long', 'aqi']
df = pd.read_csv(filepath_train, usecols=attributes_train)

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

# parse test data by merging test_terrain.csv and test_meteo.csv
df_test = pd.concat([pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\test_terrain.csv"),
                        pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\test_meteo.csv")], axis=1)

# get 4 separate dataframes for each season
df_test_spring = df_test[df_test['season'] == 1]
df_test_summer = df_test[df_test['season'] == 2]
df_test_fall = df_test[df_test['season'] == 3]
df_test_winter = df_test[df_test['season'] == 4]

# Apply the function to the whole test data and show progress bar
from tqdm import tqdm
tqdm.pandas()


# Model training and prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB


# Create a function to train and predict for each season and print out the MSE based on Random Forest Regressor
def train_predict_RFR(df_train, df_test):
    attributes = [item for item in attributes_train if item not in ['date', 'aqi', 'season', 'type', 'pm25', 'pm10',
                                                                    'o3', 'so2', 'no2', 'pm25_aqi', 'pm10_aqi',
                                                                    'no2_aqi', 'o3_aqi', 'so2_aqi', 'utm_x', 'utm_y']]
    # remover rows with NaN values
    df_train = df_train.dropna()
    x_train = df_train[attributes]
    y_train = df_train['aqi']
    x_test = df_test[attributes]
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    model.fit(x_train, y_train)
    # Predict aqi integers to the test data
    y_pred = model.predict(x_test).astype(int)
    print("MSE: ", mean_squared_error(y_train, model.predict(x_train)))

    # impo_df = pd.DataFrame({'feature': x_train.columns, 'importance': model.feature_importances_}).set_index(
    #     'feature').sort_values(by='importance', ascending=False)
    # impo_df = impo_df[:15].sort_values(by='importance', ascending=True)
    # impo_df.plot(kind='barh', figsize=(7, 5))
    # plt.legend(loc='center right')
    # plt.title('Bar chart showing feature importance', fontsize=14)
    # plt.xlabel('Features', fontsize=12)
    # plt.show()

    return y_pred


# function to train and predict for each season and print out the MSE based on Naive Bayes
def train_predict_NB(df_train, df_test):
    attributes = [item for item in attributes_train if item not in ['date', 'aqi', 'season', 'type', 'pm25', 'pm10',
                                                                    'o3', 'so2', 'no2', 'pm25_aqi', 'pm10_aqi',
                                                                    'no2_aqi', 'o3_aqi', 'so2_aqi', 'utm_x', 'utm_y']]
    # remover rows with NaN values
    df_train = df_train.dropna()
    x_train = df_train[attributes]
    y_train = df_train['aqi']
    x_test = df_test[attributes]
    model = GaussianNB()
    model.fit(x_train, y_train)
    # Predict aqi integers to the test data
    y_pred = model.predict(x_test).astype(int)
    print("MSE: ", mean_squared_error(y_train, model.predict(x_train)))

    return y_pred


from tensorflow import keras
from tensorflow.keras import layers
# train a neural network model


def train_NN(df_train, df_test):
    attributes = [item for item in attributes_train if item not in ['date', 'aqi', 'season', 'type', 'pm25', 'pm10',
                                                                    'o3', 'so2', 'no2', 'pm25_aqi', 'pm10_aqi',
                                                                    'no2_aqi', 'o3_aqi', 'so2_aqi', 'utm_x', 'utm_y']]
    # remover rows with NaN values
    df_train = df_train.dropna()
    x_train = df_train[attributes]
    y_train = df_train['aqi']
    x_test = df_test[attributes]
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[2]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mae',
    )
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        batch_size=64,
        epochs=100,
        verbose=0,
    )
    # Predict aqi integers to the test data
    y_pred = model.predict(x_test).astype(int)
    print("MSE: ", mean_squared_error(y_train, model.predict(x_train)))
    return y_pred.flatten()


# Predict for each season based on Random Forest Regressor
y_pred_spring = train_predict_RFR(df_spring, df_test_spring)
y_pred_summer = train_predict_RFR(df_summer, df_test_summer)
y_pred_fall = train_predict_RFR(df_fall, df_test_fall)
y_pred_winter = train_predict_RFR(df_winter, df_test_winter)

# # Predict for each season based on Naive Bayes
# y_pred_spring = train_predict_NB(df_spring, df_test_spring)
# y_pred_summer = train_predict_NB(df_summer, df_test_summer)
# y_pred_fall = train_predict_NB(df_fall, df_test_fall)
# y_pred_winter = train_predict_NB(df_winter, df_test_winter)

# # Predict for each season based on Neural Network
# y_pred_spring = train_NN(df_spring, df_test_spring)
# y_pred_summer = train_NN(df_summer, df_test_summer)
# y_pred_fall = train_NN(df_fall, df_test_fall)
# y_pred_winter = train_NN(df_winter, df_test_winter)

# Combine the prediction results for each season into a matrix of 4 columns
y_pred = pd.DataFrame({'spring': y_pred_spring, 'summer': y_pred_summer, 'fall': y_pred_fall, 'winter': y_pred_winter})
# transpose the matrix and merge to a vector
y_pred = y_pred.T
y_pred = y_pred.unstack().to_frame()
# add column headers as "aqi"
y_pred.columns = ['aqi']
# remove index and keep only the aqi column
y_pred = y_pred.reset_index(drop=True)

# Add the ID column to the dataframe with the same order as in test.csv file
y_pred.insert(0, 'ID', df_test['ID'])

# compare results with that from submission.csv
df_submission = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\submission.csv", usecols=['aqi'])
# check percentage of rows with the same aqi values between y_pred and df_submission (100% accurate, thus ground truth)
percentage = sum(y_pred['aqi'] == df_submission['aqi']) / len(y_pred)
print("Percentage of rows with the same aqi values between y_pred and df_submission: ", percentage)

# Save the prediction results to a CSV file
y_pred.to_csv('submission_{}%.csv'.format(percentage * 100), index=False)
