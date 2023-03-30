# Evaluating shots

## Data loading
The data are from Wyscout. We combined five leagues: Premier league, Serie A, Ligue 1, Bundesliga and LaLiga from 2021-2022 season.

## Producing datasets
make_freekickscsv.py in the folder creat_dataset create a dataset that contains all free kicks. In this dataset we added the distance, distance squared, distance cube, adjusted distance, adjusted distance squared, adjusted distance cube, aangle and arc length of shooting or crossing

## Fitting models
fit_model.py is a file that contains the fit model from free kicks of crossing and shooting.
To fit these models we used th following features:
the distance, distance squared, distance cube, adjusted distance, adjusted distance squared,
adjusted distance cube, aangle and arc length for cross and shoot.
## Plotting




