# 1st-place-GeoAI-Challenge-for-Air-Pollution
1st-place-GeoAI-Challenge-for-Air-Pollution-Susceptibility-Mapping by xiaoironman

Steps:
1. Set the current folder as your working directory.
2. Download the Test.csv file from the challenge website (https://zindi.africa/competitions/geoai-challenge-for-air-pollution-susceptibility-mapping/data) and place it in this folder as well
3. install the required packages by the commands: python -m pip install -r requirements.txt
4. run the main script: python main.py
5. You should see that the file named submission.csv is generated in the same folder, this is the one that was been submitted

Files:
1. test_terrain.csv is the terrain (topology) data extracted for the locations in each row of the Test.csv file.
2. test_meteo.csv is the meteology data extracted for the locations in each row of the Test.csv file.
3. submission.csv is the final resulted submitted to the challenge website and resulted in the #1 ranking.
4. main.py is the python script to generate the submission file
5. AI_for_Good_Air_Pollution_Challenge_Report.pdf is the report for this challenge
6. requirements.txt is the required python packages for running the main script. During development, the author used python 3.8 as the base interpreter
