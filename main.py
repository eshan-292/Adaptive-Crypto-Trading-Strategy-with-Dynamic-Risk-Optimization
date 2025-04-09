import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_orderbook_entry(entry):
    """
    Parse a single order book snapshot entry (dict) and extract:
      - timestamp
      - best bid price and volume
      - best ask price and volume
      - mid price
      - average volume (calculated from all bid and ask quantities)
    Returns a dictionary or None if data is missing.
    """
    if "bids" not in entry or "asks" not in entry:
        return None
    
    bids = entry["bids"]
    asks = entry["asks"]
    if not bids or not asks:
        return None  # no valid bids/asks

    # Best bid = highest bid price
    best_bid = max(bids, key=lambda x: x["p"])
    best_bid_price = best_bid["p"]
    best_bid_volume = best_bid["q"]
    
    # Best ask = lowest ask price
    best_ask = min(asks, key=lambda x: x["p"])
    best_ask_price = best_ask["p"]
    best_ask_volume = best_ask["q"]

    # Mid price: average of best bid and best ask prices.
    mid_price = (best_bid_price + best_ask_price) / 2

    # Calculate average volume across all bids and asks.
    avg_bid_volume = np.mean([order["q"] for order in bids])
    avg_ask_volume = np.mean([order["q"] for order in asks])
    avg_volume = (avg_bid_volume + avg_ask_volume) / 2

    return {
        "timestamp": entry["timestamp"],  # To be converted to datetime later
        "best_bid_price": best_bid_price,
        "best_ask_price": best_ask_price,
        "mid_price": mid_price,
        "best_bid_volume": best_bid_volume,
        "best_ask_volume": best_ask_volume,
        "avg_volume": avg_volume
    }

def load_orderbook_data(file_path):
    """
    Reads a data.json file where each line is one JSON snapshot.
    Returns a pandas DataFrame with:
      - timestamp (as DateTime index),
      - best_bid_price, best_ask_price, mid_price,
      - best_bid_volume, best_ask_volume, and avg_volume.
    """
    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            parsed = parse_orderbook_entry(entry)
            if parsed is not None:
                records.append(parsed)
    
    if not records:
        raise ValueError(f"No valid orderbook entries found in {file_path}")

    df = pd.DataFrame(records)
    
    # Convert 'timestamp' to DateTime (adjust the unit if needed)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def load_instrument_orderbooks(base_dir, instrument="BTC"):
    """
    Recursively walk through base_dir, find any folder named 'instrument',
    and load its 'data.json'. Concatenate all into a single DataFrame.
    """
    all_dfs = []
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == instrument:
            data_file = os.path.join(root, "data.json")
            if os.path.isfile(data_file):
                df_instrument = load_orderbook_data(data_file)
                all_dfs.append(df_instrument)
    if not all_dfs:
        raise ValueError(f"No data.json files found for {instrument} in {base_dir}")
    final_df = pd.concat(all_dfs).sort_index()
    return final_df

def load_all_instruments(base_dir):
    """
    Recursively walk through base_dir and load data.json files for ALL instruments.
    Each subfolder that has a data.json is considered an instrument folder.
    The instrument name is added as a new column.
    
    Returns:
      A single pandas DataFrame containing all instruments' data,
      with an additional 'instrument' column.
    """
    all_dfs = []
    for root, dirs, files in os.walk(base_dir):
        if 'data.json' in files:
            data_file = os.path.join(root, 'data.json')
            instrument_name = os.path.basename(root)
            df_instrument = load_orderbook_data(data_file)
            df_instrument['instrument'] = instrument_name
            all_dfs.append(df_instrument)
    if not all_dfs:
        raise ValueError(f"No data.json files found in {base_dir}")
    final_df = pd.concat(all_dfs).sort_index()
    return final_df

# Example usage:
base_directory = "sample_dataset/BINANCE_SPOT"
df_all = load_all_instruments(base_directory)

# Inspect the combined DataFrame
print(df_all.head())
print(df_all.tail())
print("Number of rows:", df_all.shape[0])
print("Unique instruments found:", df_all['instrument'].unique())


def backtest(trades, initial_capital=10000):
    """
    Compute performance metrics based on a list of trades.
    
    Parameters:
      trades         : List of trades in the form
                       (entry_time, exit_time, entry_price, exit_price, direction)
      initial_capital: Starting portfolio value.
      
    Returns:
      metrics      : Dictionary containing total return, max drawdown, Sharpe ratio,
                     win/loss ratio, and number of trades.
      equity_curve : pandas Series of the portfolio value over time (at trade exit times).
    """
    # Calculate trade returns based on trade direction.
    trade_returns = []
    for (_, _, entry_price, exit_price, direction) in trades:
        if direction == 1:  # long
            r = (exit_price - entry_price) / entry_price
        elif direction == -1:  # short
            r = (entry_price - exit_price) / entry_price
        else:
            r = 0
        trade_returns.append(r)
        
    total_return = np.prod([1 + r for r in trade_returns]) - 1

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_loss_ratio = len(wins) / len(losses) if losses else np.nan

    equity = [initial_capital]
    equity_times = []
    for (_, exit_time, _, _, _), r in zip(trades, trade_returns):
        equity.append(equity[-1] * (1 + r))
        equity_times.append(exit_time)
    equity_curve = pd.Series(equity[1:], index=equity_times)

    rolling_max = equity_curve.cummax()
    # set precision to 10 decimal places to avoid division by zero
    rolling_max = rolling_max.round(10)
    drawdown = (rolling_max - equity_curve) / rolling_max
    max_drawdown = drawdown.max()

    if len(trade_returns) > 1:
        sharpe_ratio = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(len(trade_returns))
    else:
        sharpe_ratio = np.nan

    metrics = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_loss_ratio': win_loss_ratio,
        'num_trades': len(trade_returns)
    }
    
    return metrics, equity_curve




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------
# Step 1: Enhanced Feature Engineering
# ---------------------------
def compute_technical_indicators(df, momentum_window=10):
    """
    Compute several technical indicators.
    """
    df['EMA_short'] = df['mid_price'].ewm(span=5, adjust=False).mean()
    df['EMA_long'] = df['mid_price'].ewm(span=15, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Bollinger_upper'] = df['mid_price'].rolling(window=momentum_window).mean() + 2 * df['mid_price'].rolling(window=momentum_window).std()
    df['Bollinger_lower'] = df['mid_price'].rolling(window=momentum_window).mean() - 2 * df['mid_price'].rolling(window=momentum_window).std()
    
    delta = df['mid_price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=momentum_window, min_periods=momentum_window).mean()
    avg_loss = loss.rolling(window=momentum_window, min_periods=momentum_window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['TR'] = df['mid_price'].diff().abs()
    df['ATR'] = df['TR'].rolling(window=momentum_window, min_periods=momentum_window).mean()
    
    if 'avg_volume' in df.columns:
        df['vol_MA'] = df['avg_volume'].rolling(window=momentum_window, min_periods=momentum_window).mean()
    
    df.dropna(inplace=True)
    return df

# ---------------------------
# Step 2: Generate ML Dataset with Per-Instrument Scaling
# ---------------------------
def generate_ml_dataset_per_instrument(df, lookback=20):
    """
    Generate a sequence dataset for LSTM using per-instrument scaling.
    
    For each instrument, compute technical indicators, then scale features and target separately,
    and then reassemble the data. Instrument is also one-hot encoded using a fixed list.
    
    Returns:
      X_seq: 3D numpy array of shape (num_sequences, lookback, num_features)
      y_seq: 2D numpy array of targets (num_sequences, 1)
      scaler_X_dict: Dictionary mapping instrument -> scaler for features.
      scaler_y_dict: Dictionary mapping instrument -> scaler for target.
      df_scaled: Combined scaled DataFrame (for reference).
    """
    # Compute technical indicators and target.
    df = compute_technical_indicators(df, momentum_window=10)
    df['target'] = df['mid_price'].pct_change().shift(-1)
    df.dropna(inplace=True)
    
    # Fixed list of instruments for one-hot encoding.
    instruments = ['SOL', 'USDC', 'RUNE', 'PENDLE', 'ETH', 'BTC']
    df['instrument'] = pd.Categorical(df['instrument'], categories=instruments)
    
    # We'll process each instrument separately.
    scaled_dfs = []
    scaler_X_dict = {}
    scaler_y_dict = {}
    
    # Define base feature columns.
    base_features = ['mid_price', 'EMA_short', 'EMA_long', 'MACD', 'Bollinger_upper', 'Bollinger_lower', 
                     'RSI', 'ATR']
    if 'avg_volume' in df.columns and 'vol_MA' in df.columns:
        base_features += ['avg_volume', 'vol_MA']
    
    for inst in instruments:
        df_inst = df[df['instrument'] == inst].copy()
        if df_inst.empty:
            continue
        # Create dummy variables for this instrument.
        dummies_inst = pd.get_dummies(df_inst['instrument'], prefix='inst')
        
        # Extract features and target.
        X_inst = df_inst[base_features].values
        y_inst = df_inst['target'].values.reshape(-1, 1)
        
        # Scale features and target per instrument.
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled_inst = scaler_X.fit_transform(X_inst)
        y_scaled_inst = scaler_y.fit_transform(y_inst)
        
        # Create a DataFrame for the scaled features.
        df_scaled_inst = pd.DataFrame(X_scaled_inst, columns=base_features, index=df_inst.index)
        # Append fixed dummy columns for instrument.
        df_scaled_inst = pd.concat([df_scaled_inst, dummies_inst], axis=1)
        # **Add the instrument identifier back into the DataFrame.**
        df_scaled_inst['instrument'] = inst
        
        # Add the scaled target.
        df_scaled_inst['target'] = y_scaled_inst.flatten()
        scaled_dfs.append(df_scaled_inst)
        
        # Save the scalers.
        scaler_X_dict[inst] = scaler_X
        scaler_y_dict[inst] = scaler_y
    
    # Combine all instrument data.
    df_scaled = pd.concat(scaled_dfs).sort_index()
    
    # Create sequences.
    X_all = df_scaled.drop(columns=['target']).values
    y_all = df_scaled['target'].values.reshape(-1, 1)
    sequences = []
    targets = []
    for i in range(lookback, len(X_all)):
        sequences.append(X_all[i-lookback:i])
        targets.append(y_all[i])
    
    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)
    
    return X_seq, y_seq, scaler_X_dict, scaler_y_dict, df_scaled



# ---------------------------
# Step 3: Build LSTM Model
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# Step 4: Train and Evaluate LSTM Model
# ---------------------------
def train_lstm_model(X_seq, y_seq, train_ratio=0.5, epochs=50, batch_size=32):
    n_train = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:n_train], X_seq[n_train:]
    y_train, y_test = y_seq[:n_train], y_seq[n_train:]
    
    model = build_lstm_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f"Test MSE: {mse:.6f}")
    return model, X_train, X_test, y_train, y_test

# ---------------------------
# Step 5: Resample Each Instrument Separately and Combine
# ---------------------------
resampled_dfs = []
for inst in df_all['instrument'].unique():
    df_inst = df_all[df_all['instrument'] == inst].copy()
    df_inst_resampled = df_inst.resample('10S').agg({
        'mid_price': 'last',
        'avg_volume': 'last'
    }).dropna()
    df_inst_resampled['instrument'] = inst
    resampled_dfs.append(df_inst_resampled)
df_all_processed = pd.concat(resampled_dfs).sort_index()

# ---------------------------
# Step 6: Generate ML Dataset with Per-Instrument Scaling and Train LSTM Model on Combined Data
# ---------------------------
lookback = 20
X_seq, y_seq, scaler_X_dict, scaler_y_dict, df_scaled = generate_ml_dataset_per_instrument(df_all_processed, lookback=lookback)
model_lstm, X_train, X_test, y_train, y_test = train_lstm_model(X_seq, y_seq, train_ratio=0.5, epochs=50, batch_size=32)
# save the model 
model_lstm.save('lstm_model.h5')
# save the scalar y
import joblib
joblib.dump(scaler_y_dict, 'scaler_y_dict.pkl')

# load the saved model 

# ---------------------------
# Step 7: Evaluate the LSTM Model on a Specific Instrument (e.g., ETH)
# ---------------------------
def ml_strategy_lstm_per_instrument(df, model, scaler_y, lookback=20, 
                                    long_threshold=0.001, short_threshold=-0.001):
    """
    Use the trained LSTM model to generate trade signals on a single instrument DataFrame.
    This function assumes that df corresponds to a single instrument.
    """
    # Generate dataset (without re-scaling per instrument, since we already scaled during training).
    X_seq, y_seq, _, _, df_features = generate_ml_dataset_per_instrument(df, lookback=lookback)
    predictions_scaled = model.predict(X_seq)
    # Inverse-transform predictions using the scaler for that instrument.
    # We need to get the instrument name (assume all rows have the same instrument).
    instrument = df_features['instrument'].iloc[0]
    # Get the target scaler for that instrument.
    # Here we assume you have stored the scaler for each instrument in scaler_y_dict.
    # If not, you could fit a new scaler on df for that instrument.
    predicted_returns = scaler_y[instrument].inverse_transform(predictions_scaled)
    
    df_signals = df_features.iloc[lookback:].copy()
    df_signals['predicted_return'] = predicted_returns.flatten()
    
    df_signals['position'] = 0
    df_signals.loc[df_signals['predicted_return'] > long_threshold, 'position'] = 1
    df_signals.loc[df_signals['predicted_return'] < short_threshold, 'position'] = -1
    
    trades = []
    current_position = 0
    entry_time = None
    entry_price = None
    for time, row in df_signals.iterrows():
        new_position = row['position']
        if current_position == 0 and new_position != 0:
            current_position = new_position
            entry_time = time
            entry_price = row['mid_price']
        elif current_position != 0 and new_position != current_position:
            exit_time = time
            exit_price = row['mid_price']
            trades.append((entry_time, exit_time, entry_price, exit_price, current_position))
            current_position = new_position
            if current_position != 0:
                entry_time = time
                entry_price = row['mid_price']
    if current_position != 0:
        exit_time = df_signals.index[-1]
        exit_price = df_signals.iloc[-1]['mid_price']
        trades.append((entry_time, exit_time, entry_price, exit_price, current_position))
    
    df_signals['strategy_return'] = df_signals['mid_price'].pct_change() * df_signals['position'].shift(1)
    df_signals['strategy_equity'] = (1 + df_signals['strategy_return']).cumprod()
    initial_capital = 10000
    equity_curve = initial_capital * df_signals['strategy_equity']
    
    return trades, equity_curve, df_signals

# # Test on ETH (for which we assume we already have a scaler in scaler_y_dict).
# df_ETH = df_all_processed[df_all_processed['instrument'] == 'ETH']
# trades_eth, equity_curve_eth, df_ml_eth = ml_strategy_lstm_per_instrument(df_ETH.copy(), model_lstm, scaler_y_dict, lookback=lookback,
#                                                                            long_threshold=0.001, short_threshold=-0.001)
# metrics_eth, _ = backtest(trades_eth, initial_capital=10000)
# print("LSTM ML Strategy Performance Metrics for ETH:")
# print(metrics_eth)

# plt.figure(figsize=(12, 6))
# plt.plot(equity_curve_eth.index, equity_curve_eth, label='Equity Curve (ETH LSTM Strategy)')
# plt.title('LSTM ML Strategy Equity Curve for ETH')
# plt.xlabel('Time')
# plt.ylabel('Equity')
# plt.legend()
# plt.grid(True)
# plt.show()


# List of instruments from the fixed list we use.
instruments = ['SOL', 'USDC', 'RUNE', 'PENDLE', 'ETH', 'BTC']

# Dictionary to store performance metrics for each instrument.
results = {}

for inst in instruments:
    print(f"Processing instrument: {inst}")
    # Filter data for the instrument
    df_inst = df_all_processed[df_all_processed['instrument'] == inst].copy()
    # If there is not enough data, skip this instrument.
    if df_inst.empty or len(df_inst) < 50:
        print(f"Not enough data for {inst}, skipping.")
        continue

    # Run the ML strategy for the instrument using our per-instrument dataset.
    trades_inst, equity_curve_inst, df_signals_inst = ml_strategy_lstm_per_instrument(
        df_inst.copy(),
        model_lstm,
        scaler_y_dict,  # scaler_y_dict is our dictionary with target scalers per instrument
        lookback=lookback,
        long_threshold=0.001,
        short_threshold=-0.001
    )
    
    # Compute performance metrics for the instrument.
    metrics_inst, _ = backtest(trades_inst, initial_capital=10000)
    results[inst] = metrics_inst
    print(f"Results for {inst}: {metrics_inst}")

# Convert results to DataFrame and save to CSV.
results_df = pd.DataFrame(results).T
results_df.index.name = 'instrument'
results_df.to_csv('ml_strategy_results.csv')

print("Saved per-instrument performance metrics to ml_strategy_results.csv")
