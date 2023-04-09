import pandas as pd


def bollinger_bands(df, price='close', period=20, deviation=1.8):
    # copy the original dataframe
    df_copy = df.copy()
    
    # calculate the moving average
    df_copy['bb_middle'] = df_copy[price].rolling(window=period).mean()
    
    # calculate the standard deviation
    df_copy['bb_std'] = df_copy[price].rolling(window=period).std()
    
    # calculating the upper and lower borders of Bollinger Bands
    df_copy['bb_upper'] = df_copy['bb_middle'] + (df_copy['bb_std'] * deviation)
    df_copy['bb_lower'] = df_copy['bb_middle'] - (df_copy['bb_std'] * deviation)
    
    # deleting auxiliary column with a standard deviation
    df_copy.drop(columns=['bb_std'], inplace=True)

    return df_copy

def rsi(df, price='close', period=14):
    # copy the original dataframe
    df_copy = df.copy()
    
    # calculate price defference
    delta = df_copy[price].diff()
    
    # calculate the rise and fall of prices 
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculation of average values of increase and decrease in prices for the period 
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Relative Strength Calculation (RS)
    rs = avg_gain / avg_loss
    
    # Relative Strength Index Calculation (RSI)
    df_copy['rsi'] = 100 - (100 / (1 + rs))

    return df_copy