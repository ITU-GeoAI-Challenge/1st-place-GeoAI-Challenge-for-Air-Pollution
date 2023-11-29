import pandas as pd


filepath_train = r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\Train.csv"
filepath_test = r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\Test.csv"

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

# parse the file containing air pollution timeseries data and Digital terrain model data
df_terrain = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\air_pollution_Milan_Comune_topo_only.csv")


# function to get the proper row of terrain data by finding the closest point to the input coordinates
def get_terrain_data(lat, lng):
    # find the closest point to the input coordinates
    df_terrain['distance'] = df_terrain.apply(lambda row: (row['lat'] - lat) ** 2 + (row['lng'] - lng) ** 2, axis=1)
    df_terrain.sort_values(by=['distance'], inplace=True)
    # get the closest point
    df_terrain_closest = df_terrain.iloc[0]
    # get the terrain data of the closest point
    df_terrain_data = df_terrain_closest.iloc[5:]
    return df_terrain_data


# Read test data from CVS file with only the columns of ['season', 'lat', 'lng']
df_test = pd.read_csv(filepath_test, usecols=['season', 'lat', 'lng', 'ID'])
# Apply the function to every row of test data and show progress bar
df_test_terrain = df_test.progress_apply(lambda row: get_terrain_data(row['lat'], row['lng']), axis=1)

# combine df_test and df_test_terrain
df_test_terrain = pd.concat([df_test, df_test_terrain], axis=1)
# save the test data to a csv file
df_test_terrain.to_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\test_terrain.csv")

# parse meteorological data for each season
df_meteo_spring = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\meteo_winter_2022.csv")
df_meteo_summer = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\meteo_spring_2022.csv")
df_meteo_fall = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\meteo_summer_2022.csv")
df_meteo_winter = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\meteo_autumn_2022.csv")


# function to get the proper row of meteo data by finding the closest point to the input coordinates
def get_meteo_data(df_meteo, lat, lng):
    # find the closest point to the input coordinates
    df_meteo['distance'] = df_meteo.apply(lambda row: (row['lat'] - lat) ** 2 + (row['lng'] - lng) ** 2, axis=1)
    df_meteo.sort_values(by=['distance'], inplace=True)
    # get the closest point
    df_meteo_closest = df_meteo.iloc[0]
    # get the meteo data of the closest point
    df_meteo_data = df_meteo_closest.iloc[3:-5]
    return df_meteo_data


# Apply the function to every row of test data and show progress bar
df_test_meteo_spring = df_test_spring.progress_apply(lambda row: get_meteo_data(df_meteo_spring, row['lat'], row['lng']), axis=1)
df_test_meteo_summer = df_test_summer.progress_apply(lambda row: get_meteo_data(df_meteo_summer, row['lat'], row['lng']), axis=1)
df_test_meteo_fall = df_test_fall.progress_apply(lambda row: get_meteo_data(df_meteo_fall, row['lat'], row['lng']), axis=1)
df_test_meteo_winter = df_test_winter.progress_apply(lambda row: get_meteo_data(df_meteo_winter, row['lat'], row['lng']), axis=1)

# combine the 4 dataframes of test meteo data into one dataframe row by row
df_test_meteo = pd.concat([df_test_meteo_spring, df_test_meteo_summer, df_test_meteo_fall, df_test_meteo_winter], axis=0)
df_test_meteo.sort_index(inplace=True)
# save the test data to a csv file
df_test_meteo.to_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\test_meteo.csv")


# plot the aqi distribution on a map for each season using plotly
import plotly.express as px


def plot_map(df, season):
    # scatter plot with color bar range from 0 to 10
    fig = px.scatter_mapbox(df, lat="lat", lon="lng", color="aqi", size="aqi", hover_data=['aqi'],
                            color_continuous_scale=px.colors.sequential.Plasma, zoom=10, range_color=[0, 8])

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title_text="Air Quality Index in " + season)

    # font size of the title text
    fig.update_layout(title_font_size=25)
    # font size of the color bar text
    fig.update_layout(coloraxis_colorbar=dict(
        title="AQI",
        tickfont={'size': 25},
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=1
    ))

    fig.show()


def map_train():
    plot_map(df_spring[df_spring['date'] == '2016-02-15'], 'Spring')
    plot_map(df_summer[df_summer['date'] == '2016-05-15'], 'Summer')
    plot_map(df_fall[df_fall['date'] == '2016-08-15'], 'Fall')
    plot_map(df_winter[df_winter['date'] == '2016-11-15'], 'Winter')


def map_test():
    # load test data
    df_test = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\Test.csv")
    # load submission file
    df_submission = pd.read_csv(r"C:\Users\Xiao Liu\Downloads\Zindi_air_pollution\submission.csv")
    # combine test data and submission file
    df_test_submission = pd.merge(df_test, df_submission, on='ID', how='left')
    # plot the aqi distribution on a map for each season using plotly
    plot_map(df_test_submission[df_test_submission['season'] == 1], 'Spring')
    plot_map(df_test_submission[df_test_submission['season'] == 2], 'Summer')
    plot_map(df_test_submission[df_test_submission['season'] == 3], 'Fall')
    plot_map(df_test_submission[df_test_submission['season'] == 4], 'Winter')