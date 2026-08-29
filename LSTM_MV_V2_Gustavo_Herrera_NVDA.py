# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:10:47 2026

@author: herrerag
"""
#%%
"""
IMPORTANT:Multivariate model. This file is for changing the parameters. Record
the results before trying a new one. Pay attention to the configuration if
you remove or add variables

"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


#%% Define Data Directory

data_dir = os.getcwd()   # get the current working directory where the script is running (optional)

#%% Reproducibility

Seed = 13

os.environ["PYTHONHASHSEED"] = str(Seed)
random.seed(Seed)
np.random.seed(Seed)
tf.random.set_seed(Seed)

#%% Load TAQ Realized Volatility Data

df = pd.read_pickle(os.path.join(data_dir, "daily_metrics.pkl"))

print("Raw dataset info:")
print(df.info())
print(df.head())

#%% Select Asset (NVDA)

nvda = df[df['sym_root'] == 'NVDA'].copy() # filter the dataset to keep only rows where the asset ticker is NVDA
print(nvda.head())

#%% Pivot Realized Volatility Panel

df_pivot = df.pivot(
    index='date',
    columns='sym_root',  # columns become different stock tickers
    values='realized_volatility'
)

print("Pivot table info:")
print(df_pivot.info())

# Reset index for merge
df_pivot = df_pivot.reset_index()

df_pivot['date'] = pd.to_datetime(df_pivot['date'])

#%% Load VIX Data

vix = pd.read_csv(os.path.join(data_dir, "VIXCLS.csv"))   
vix['date'] = pd.to_datetime(vix['observation_date'])     # convert observation_date to datetime
vix = vix[['date','VIXCLS']]                               # keep only date and VIX value

df.info()                                                  # check original dataframe info

df_pivot.info()                                            # check pivot dataframe structure

#%% Join Two Files

df_pivot = df_pivot.set_index('date')                      # set date as index
vix = vix.set_index('date')                                # set date as index for VIX
data = df_pivot[['NVDA','QQQ', "TSM"]].join(vix)             # join VIX to realized volatility using index

data.head()  

#%% Convert Data Types
data['VIXCLS'] = pd.to_numeric(data['VIXCLS'], errors='coerce')

#%% Multivariate model

data_multi = data[["NVDA", "QQQ", "TSM", "VIXCLS"]].copy()
data_multi["NVDA"] = data_multi["NVDA"] * 100 #scaling
#data_multi["QQQ"] = data_multi["QQQ"] * 100 #scaling
#data_multi["TSM"] = data_multi["TSM"] * 100 #scaling


#Check missing values 
print(data_multi.isna().sum())  #No missing values
#%% Log transformation since the data is right skewed and only positive

data_multi["Log_NVDA"]=np.log(data_multi["NVDA"])
#data_multi["Log_QQQ"]=np.log(data_multi["QQQ"])
#data_multi["Log_TSM"]=np.log(data_multi["TSM"])
data_multi["Log_VIX"]=np.log(data_multi["VIXCLS"])


# plt.hist(data_multi["Log_NVDA"],bins=20) #Just to confirm
# plt.title("Realized Volatility (LOG) - NVDA")
# plt.show()
# plt.hist(data_multi["Log_QQQ"], bins=20) #Just to confirm
# plt.title("Realized Volatility (LOG) - QQQ")
# plt.show()
# plt.hist(data_multi["Log_TSM"], bins=20) #Just to confirm
# plt.title("Realized Volatility (LOG) - TSM")
# plt.show()
# plt.hist(data_multi["Log_VIX"], bins=20) #Just to confirm 
# plt.title("Implied Volatility (LOG) - VIX")
# plt.show()

#%% Final data set

multi_features = data_multi[["Log_NVDA","Log_VIX"]].copy()

#%% Chronological train (70%), validation (20%), test (10%) split

n = len(multi_features)

train_n= int(n*.70)
valid_n= int(n*.90)

train_df= multi_features.iloc[:train_n].copy()
valid_df= multi_features.iloc[train_n:valid_n].copy()
test_df= multi_features.iloc[valid_n:].copy()

print("Sample sizes:")
print("Train:", len(train_df))
print("Validation:", len(valid_df))
print("Test:", len(test_df))

print("Date ranges:")
print("Train:", train_df.index.min(), "to", train_df.index.max())
print("Validation:", valid_df.index.min(), "to", valid_df.index.max())
print("Test:", test_df.index.min(), "to", test_df.index.max())

#%% Data standardization

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_df)
valid_scaled = scaler.transform(valid_df)
test_scaled  = scaler.transform(test_df)

#%% Convert back to data frames and preserve the dates

train_scaled = pd.DataFrame(train_scaled, index=train_df.index, columns=train_df.columns)
valid_scaled = pd.DataFrame(valid_scaled, index=valid_df.index, columns=valid_df.columns)
test_scaled  = pd.DataFrame(test_scaled, index=test_df.index, columns=test_df.columns)

#%% Define sequence length
# Start with 21 trading days, about 1 trading month
# This is a reasonable first choice, then it could be increased if needed

seq_length = 15
batch_size = 32

#%% Helper function to create dataset
def make_tf_dataset(features_df, seq_length, batch_size, shuffle=False):
    """
    features_df must include:
    Log_NVDA, Log_QQQ, Log_TSM, Log_VIX

    X = past seq_length days of all features
    y = next-day Log_NVDA only
    """
    data_array = features_df.to_numpy(dtype=np.float32)
    
    target_idx = features_df.columns.get_loc("Log_NVDA")
    targets = data_array[seq_length:, target_idx] #Start on the seq_lenght row

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data_array,
        targets=targets,               
        sequence_length=seq_length,           # length of each sequence
        sequence_stride=1,                    # move one day at a time (default)
        sampling_rate=1,                      # use every observation (default)
        batch_size=batch_size,
        shuffle=shuffle
    )
    return ds
#%% Build data sets

train_ds = make_tf_dataset(train_scaled, seq_length, batch_size, shuffle=False)
valid_ds = make_tf_dataset(valid_scaled, seq_length, batch_size, shuffle=False)
test_ds = make_tf_dataset(test_scaled, seq_length, batch_size, shuffle=False)

#%% Clear session
tf.keras.backend.clear_session()
tf.random.set_seed(Seed)

#%% Multi-layer LSTM model 

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_length, 2)), #X time steps, X inputs
    tf.keras.layers.LSTM(8, return_sequences=True), #return_sequences=True is required to pass the full sequence to the next LSTM layer
    tf.keras.layers.LSTM(8), 
    tf.keras.layers.Dense(1)                     # 1 dense layer
])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse",
    patience=20,
    restore_best_weights=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=optimizer,
    metrics=["mae", "mse"]
)

#%% Train model

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=500,
    callbacks=[early_stopping_cb],
    verbose=1
)

history_df = pd.DataFrame(history.history)

#%% Plot history
plt.figure(figsize=(10, 5))

# Loss (MSE)
plt.plot(history_df.index + 1, history_df["loss"], label="Train Loss (MSE)")
plt.plot(history_df.index + 1, history_df["val_loss"], label="Validation Loss (MSE)")

# Optional: MAE (only if you included it in compile)
if "mae" in history_df.columns:
    plt.plot(history_df.index + 1, history_df["mae"], linestyle="--", label="Train MAE")
    plt.plot(history_df.index + 1, history_df["val_mae"], linestyle="--", label="Validation MAE")

plt.title("Multi-Layer LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.xlim([1, 500])   # adjust if needed
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Helper functions for inverse transformation
# Separate scaler for target only
target_scaler = StandardScaler()
target_scaler.fit(train_df[["Log_NVDA"]])

def evaluate_helper(model, ds, original_df, target_scaler, seq_length, label):
    
    # Predictions (scaled log space)
    y_pred_scaled = model.predict(ds, verbose=0).reshape(-1, 1)
    
    # Inverse scaling (back to log space)
    y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    
    # True values (log space, aligned)
    y_true_log = original_df["Log_NVDA"].iloc[seq_length:].to_numpy().reshape(-1, 1)
    
    # Convert to original scale
    y_pred = np.exp(y_pred_log).flatten()
    y_true = np.exp(y_true_log).flatten()
    
    # Keep aligned dates for plotting
    dates = original_df.index[seq_length:]
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{label} MAE:  {mae:.6f}")
    print(f"{label} RMSE: {rmse:.6f}")
    
    # Return everything needed for plotting
    results = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    }, index=dates)
    
    return results, mae, rmse

#%% Evaluate training and validation results
train_results, train_mae, train_rmse = evaluate_helper(
    model, train_ds, train_df, target_scaler, seq_length, "Train")

valid_results, valid_mae, valid_rmse = evaluate_helper(
    model, valid_ds, valid_df, target_scaler, seq_length, "Validation")

test_results, test_mae, test_rmse = evaluate_helper(
    model, test_ds, test_df, target_scaler, seq_length, "Test")

#%% Plot predictions vs actuals - Train
plt.figure(figsize=(12, 5))
plt.plot(train_results.index, train_results["Actual"], label="Actual")
plt.plot(train_results.index, train_results["Predicted"], label="Predicted")
plt.title("Train Set: Actual vs Predicted Realized Volatility")
plt.xlabel("Date")
plt.ylabel("Realized Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot predictions vs actuals - Validation
plt.figure(figsize=(12, 5))
plt.plot(valid_results.index, valid_results["Actual"], label="Actual")
plt.plot(valid_results.index, valid_results["Predicted"], label="Predicted")
plt.title("Validation Set: Actual vs Predicted Realized Volatility")
plt.xlabel("Date")
plt.ylabel("Realized Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% All predictions plotted
plt.figure(figsize=(12, 5))

plt.plot(train_results.index, train_results["Actual"], label="Train Actual", alpha=0.7)
plt.plot(train_results.index, train_results["Predicted"], label="Train Predicted", alpha=0.7)

plt.plot(valid_results.index, valid_results["Actual"], label="Validation Actual", alpha=0.7)
plt.plot(valid_results.index, valid_results["Predicted"], label="Validation Predicted", alpha=0.7)

plt.plot(test_results.index, test_results["Actual"], label="Test Actual", linewidth=2)
plt.plot(test_results.index, test_results["Predicted"], label="Test Predicted", linewidth=2)

plt.title("Actual vs Predicted Realized Volatility (All Sets)")
plt.xlabel("Date")
plt.ylabel("Realized Volatility (x100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Only on testing data set
plt.figure(figsize=(12, 5))

plt.plot(test_results.index, test_results["Actual"], label="Actual", linewidth=2)
plt.plot(test_results.index, test_results["Predicted"], label="Predicted", linewidth=2)

plt.title("Test Set: Actual vs Predicted Realized Volatility")
plt.xlabel("Date")
plt.ylabel("Realized Volatility (x100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()













