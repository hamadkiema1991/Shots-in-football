# Evaluating shots

## Data loading
The data are from Wyscout. We combined five leagues: Premier league, Serie A, Ligue 1, Bundesliga and LaLiga from 2021-2022 season.

## Producing datasets
make_freekickscsv.py in the creat_dataset folder: create a dataset that contains all free kicks. In this dataset we added the distance, distance squared, distance cube, adjusted distance, adjusted distance squared, adjusted distance cube, aangle and arc length of shooting or crossing

## Fitting models
fit_model.py is loacated in the freekick_analysis folder: is a file that contains the fit model from free kicks of crossing and shooting.
To fit these models we used th following features:
the distance, distance squared, distance cube, adjusted distance, adjusted distance squared,
adjusted distance cube, aangle and arc length for cross and shoot.
## Plotting
1. data_histogram.py is loacated in the freekick_analysis folder:
 1.1 Number_of_shoots_from_free_kick.png: is the number of shooting from free kicks  from           differents points on the pitch.
 1.2 Number_of_crosses_from_free_kick.png:is the number of crossing from free kicks from           differents points on the pitch.
 1.3 Number_of_shots_minus_of_crosses_from free kick.png: is the number of shooting minus             crossing from free kicks from  differents points on the pitch
2. plot_curves.py is loacated in the freekick_analysis folder:



