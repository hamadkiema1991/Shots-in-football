# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:41:19 2023

@author: hamad
"""
import pandas as pd
import numpy as np
import pitch
import matplotlib.pyplot as plt
#import machine learning libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('../data/freeKicks.csv')

columns = ['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                    'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length']


freeKickShot=df[df['free_kick_type']=='free_kick_shot']




freeKickCross=df[df['free_kick_type']=='free_kick_cross']
#construct the feature matrix and the target variable
X = freeKickShot[columns]
y = freeKickShot['goal']



#spllit the data to train, validation and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 123, stratify = y)
X_cal, X_val, y_cal, y_val  = train_test_split(X_test, y_test, train_size = 0.5, random_state = 123, stratify = y_test)

#scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_cal = scaler.transform(X_cal)

#creating a function with a model architecture
def create_model():
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation = 'sigmoid'),
    ])
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss="mean_squared_error" , metrics=['accuracy'])
    return model

#create model
model = create_model()

#create an early stopping object
callback = EarlyStopping(min_delta=1e-5, patience = 50, mode = "min", monitor = "val_loss", restore_best_weights=True)

#fit the model 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=1, batch_size=16, callbacks = [callback])    

fig, axs = plt.subplots(2, figsize=(10,12))
#plot training history - accuracy
axs[0].plot(history.history['accuracy'], label='train')   
axs[0].plot(history.history['val_accuracy'], label='validation')
axs[0].set_title("Accuracy at each epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

#plot training history - loss function
axs[1].plot(history.history['loss'], label='train')   
axs[1].plot(history.history['val_loss'], label='validation')
axs[1].legend()
axs[1].set_title("Loss at each epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE")
plt.show()

##############################################################################
# Assessing our model
# ----------------------------
# To assess our model, we calculate ROC AUC and investigate calibration curves. From the plots we can see that some of higher probabilities are
# underestimated by our model, but these are satisfactory results given the number of data we have and a shallow network. Also, we calculate Brier score
# on unseen data. It amounts to 0.08, which is a good score. 

#ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
fig, axs = plt.subplots(2, figsize=(10,12))
y_pred = model.predict(X_cal)
fpr, tpr, _ = roc_curve(y_cal,  y_pred)
auc = roc_auc_score(y_cal, y_pred)
axs[0].plot(fpr,tpr,label= "AUC = " + str(auc)[:4])
axs[0].plot([0, 1], [0, 1], color='black', ls = '--')
axs[0].legend()
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_title('ROC curve')

#CALIBRATION CURVE
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_cal, y_pred, n_bins=10)
axs[1].plot(prob_true, )
axs[1].plot([0, 1], [0, 1], color='black', ls = '--')
axs[1].set_ylabel('Empirical Probability')
axs[1].set_xlabel('Predicted Probability')
axs[1].set_title("Calibration curve")
plt.show()
#Brier score
print("Brier score", brier_score_loss(y_cal, y_pred))


#store data in a matrix
X_unseen = freeKickShot[columns]

#scale data
X_uns = scaler.transform(X_unseen)

#make predictions
xgs_shot= model.predict(X_uns)

#find  xG
freeKickShot["our_xG"] = xgs_shot



#Create a 2D map of xG
pgoal_2d_shot=np.zeros((68,68))
for x in range(68):
    for y in range(68):
        
# We divide the  penalty area to three sections: to the left, above of penalty area and to the right.
# In each section, we create a data frame for every point and then to apply our model 
# to predict the probability to score

      # Compute probability of goal to above of penalty area 
      if (y>14) & (y<=53)& (x>=16.5):
          # in this section we create a data frame 
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          
          #distance squared
          distance_squared=np.power(distance,2)
          
          #distance cube
          distance_cube=np.power(distance,3)
          
          #adjusted distance
          adj_distance=abs(distance-16.5)
          
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #scale data
          X_uns = scaler.transform(xG)
          #make predictions
          xg= model.predict(X_uns)*100
          pgoal_2d_shot[x,y]=xg
        # Compute probability of goal to the right of penalty area     
      if (y>53):
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #scale data
          X_uns = scaler.transform(xG)
          #make predictions
          xg= model.predict(X_uns)*100
          pgoal_2d_shot[x,y]=xg
      
     # Compute probability of goal to the left of penalty area 
      if (y<=14):
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #scale data
          X_uns = scaler.transform(xG)
          #make predictions
          xg= model.predict(X_uns)*100
          pgoal_2d_shot[x,y]=xg




#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal_2d_shot, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=10, zorder = 1)
fig.colorbar(pos, ax=ax)


