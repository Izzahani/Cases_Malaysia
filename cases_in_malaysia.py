# %% Import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pickle

# %% Step 1) Data Loading
CSV_FILE = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
df_train = pd.read_csv(CSV_FILE)

# %% Step 2) Data Inspection
df_train.head()

# %%
df_train.isnull().sum()

# %%
df_train.info()

# %% Step 3) Data Cleaning
# to convert into numerical numbers
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')

df_train.info()

# %%
df_train.isnull().sum()

# %%
plt.figure(figsize=(10,10))
plt.plot(df_train['cases_new'])
plt.show()
# %% to fill the NaN
df_train['cases_new'] = df_train['cases_new'].interpolate(method='polynomial', order=2)

df_train['cases_new'] = df_train['cases_new'].astype('int64')

df_train.info()

# %%
df_train.isnull().sum()
# %%
plt.figure(figsize=(10,10))
plt.plot(df_train['cases_new'])
plt.show()
# %% Step 4) Features Selection

# %% Step 5) Data Pre-processing
# to get the numpy array
data = df_train['cases_new'].values
data = data[::, None]

# %% to convert to 0 - 1
mm_scaler = MinMaxScaler()
mm_scaler.fit(data)
data = mm_scaler.transform(data)

# %% Data Development
win_size = 30
X_train = []
y_train = []

for i in range(win_size, len(data)): #range(30,end)
    X_train.append(data[i-win_size:i]) #[0:30]
    y_train.append(data[i])

# to convert into numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)
# %%
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=123)
# %% Data Evaluation
model = Sequential()
model.add(Input(shape=(win_size,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(32, return_sequences=True)) 
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dense(1, activation= 'linear'))
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse','mape']) 

# %%
plot_model(model, show_shapes = True)

#%% callbacks
#early stopping and tensorboard
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

hist = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_callback, early_stop_callback], validation_data=(X_test, y_test))

# %% Step 1) Load Testing Data
TEST_CSV = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')
df_test = pd.read_csv(TEST_CSV)

# %% Step 2) Data Inspection
df_test.info()

# %%
df_test.isnull().sum()

# %% Step 3) Data Cleaning
# to fill the NaN
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial', order=2)
# %%
# convert to integer
df_test['cases_new'] =  df_test['cases_new'].astype('int64')
df_test.isnull().sum()

# %% Step 4) Data Preprocessing
# combine train and test
concat = pd.concat((df_train['cases_new'], df_test['cases_new']))
concat = concat[len(concat) - win_size - len(df_test):] 

concat = mm_scaler.transform(concat[::, None])

X_test = []
y_test = []

for i in range(win_size, len(concat)):
    X_test.append(concat[i-win_size:i])
    y_test.append(concat[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

# %% to predict the stock price based on the testing dataset
predicted_price = model.predict(X_test)

# %% To visualize predicted stock price and actual price

plt.figure()
plt.plot(predicted_price, color='red')
plt.plot(y_test, color='blue')
plt.legend(['Predicted', 'Actual'])
plt.ylabel('Cases')
plt.show()

y_test = mm_scaler.inverse_transform(y_test)
predicted_price = mm_scaler.inverse_transform(predicted_price)

plt.figure()
plt.plot(predicted_price, color='red')
plt.plot(y_test, color='blue')
plt.legend(['Predicted', 'Actual'])
plt.ylabel('Cases')
plt.show()

#metrics to evaluate the performance
print("MAPE: " + str(mean_absolute_percentage_error(y_test, predicted_price)))
print("MAE: " + str(mean_absolute_error(y_test, predicted_price)))

# %% Model Saving
# save min max scaler
with open('mm_scaler.pickle', 'wb') as f:
    pickle.dump(mm_scaler, f)

# save model
model.save('model.h5')