# __Evaluating shots__

## __Data loading__
The data are from __Wyscout__. We combined five leagues: Premier league, Serie A, Ligue 1, Bundesliga and LaLiga from 2021-2022 season.

## __Producing datasets__
The python file __make_freekickscsv.py__ in the folder creat_dataset create a dataset that contains all free kicks. In this dataset we added the distance, distance squared, distance cube, adjusted distance, adjusted distance squared, adjusted distance cube, aangle and arc length of shooting or crossing


## __Fitting models__
In __fit_model.py__ file we fitted three models using logistic regression to determine the probability of scoring at different locations on the pitch: free kicks by shooting, by crossing and by open play. In addition, we plotted the contour of each of them. That file is in freekick_analysis folder.


## __Figures__
1.__contour_plot_freeKicks_shot.png__: represents the free kick from shoot contour. It shows the probabitlity of scoring at different locations on the pitch. Her python code is in fit_model.py file.

2.__contour_plot_freeKick_cross.png__: represents the free kick from cross contour. It shows the probabitlity of scoring at different locations on the pitch. Her python code is in fit_model.py file.

3.__contour_plot_freeKicks_Open_play.png__: represents the free kick from open play contour. It shows the probabitlity of scoring at different locations on the pitch. Her python code is in fit_model.py file.


4.__contour_plot_Open_play_freeKicks_and_Cross.png__: represents the free kick from open play and cross contour. It indicates the part of the pitch where a cross or open play is to be made. Her python code is fit_model.py file.

5.__contour_plot_Open_play_freeKicks_and_shot.png__: represents the free kick from open play and shot contour. It indicates the part of the pitch where a shot or open play is to be made. Her python code is in fit_model.py file.

6.__what_to_do.png__: represents the free kick from open play, shoot and cross contour. It indicates the part of the pitch where a shoot or open play or cross is to be made. Her python code is in fit_model.py file.

7.__Number_of_shoots_from_free_kick.png__: represents the number of free kicks from shoot. That figure is a heatmap. Her python code is in data_histogram.py file. This file is in freekick_analysis folder.

8.__Number_of_crosses_from_free_kick.png__: represents the number of free kicks from cross. That figure is a heatmap. Her python code is in data_histogram.py file. This file is in freekick_analysis folder.

9.__Number_of_shots_minus_of_crosses_from_free_kick.png__: represents the number of free kicks from shoot minus from cross. That figure is a heatmap. Her python code is in data_histogram.py file. This file is in freekick_analysis folder.






