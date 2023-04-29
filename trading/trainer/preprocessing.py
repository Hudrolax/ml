import numpy as np
import pandas as pd
from datetime import datetime


def minmax_scale_array_global(arr:np.ndarray | pd.DataFrame, feature_range=(0, 1)):
    """
    Scales a 2D numpy array using a global minimum and maximum.
    
    Parameters:
    arr (numpy.ndarray | pd.DataFrame): Input 2D array or Dataframe.
    feature_range (tuple): Tuple containing two numbers (min, max), defining the scaling range.
    
    Returns:
    numpy.ndarray: Scaled 2D array.
    """
    input_array = True
    cols = []
    if isinstance(arr, pd.DataFrame):
        cols = arr.columns
        arr = arr.values
        input_array = False

    if not isinstance(arr, np.ndarray) or len(arr.shape) != 2:
        raise ValueError("Input array must be a 2D numpy array.")
    
    min_val, max_val = feature_range
    global_min = np.min(arr)
    global_max = np.max(arr)
    
    if global_max - global_min != 0:
        scaled_arr = (arr - global_min) * (max_val - min_val) / (global_max - global_min) + min_val
        return scaled_arr if input_array else pd.DataFrame(scaled_arr, columns=cols)
    else:
        return arr if input_array else pd.DataFrame(arr, columns=cols) 

def extend_dataframe(df, window):
    """
    Extends the dataframe to the given window size by adding rows with the last value in each column.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        window (int): Desired number of rows in the output dataframe.

    Returns:
        pd.DataFrame: Extended dataframe.
    """
    # Calculate the number of rows to add
    rows_to_add = window - len(df)

    # Check if any rows need to be added
    if rows_to_add > 0:
        # Get the last row values
        last_row_values = df.iloc[-1]

        # Create a list to store the new rows
        new_rows = []

        for _ in range(rows_to_add):
            # Add the last row values to the list
            new_rows.append(last_row_values)

        # Create a new dataframe with the same column names and 'rows_to_add' number of rows
        new_rows_df = pd.DataFrame(new_rows, columns=df.columns)

        # Concatenate the input dataframe with the new rows dataframe
        df_extended = pd.concat([df, new_rows_df], ignore_index=True)
        
        return df_extended
    else:
        return df

def make_observation_window(dataframes:list[pd.DataFrame], date:datetime, window:int, columns:list=[],
                            divider:bool = False) -> pd.DataFrame:
    # 1. Проверка наличия поля 'date' во всех датафреймах
    for df in dataframes:
        exist_date = False
        for col in df.columns:
            if 'date' in col:
                exist_date = True

        if not exist_date:
            raise ValueError(f"All dataframes must contain 'date' column.")
    
    date_col = 'date'
    for col in dataframes[0].columns:
        if 'date' in col:
            date_col = col
    
    # 3. Выбор начальной даты
    start_date = dataframes[0][dataframes[0][date_col] < date][date_col].max()
    
    # 4. Выбор window-строк из каждого датафрейма
    selected_data = []
    for df in dataframes:
        # get prefix
        prefix = ''
        if '-' in df.columns[0]:
            name = df.columns[0].split('-')
            prefix = name[0] + '-'

        # select window-data from df
        temp = df[df[prefix + 'date'] <= start_date].iloc[-window:].copy().reset_index(drop=True)
        temp = extend_dataframe(temp, window)

        # get column names with prefix
        _columns = []
        for col in columns:
            _columns.append(prefix + col)

        # pick columns
        if len(_columns) > 0:
            temp = temp[_columns]
        else:
            temp.drop(columns=[prefix + 'date'], inplace=True)
            _columns = temp.columns

        temp.index = pd.RangeIndex(start=window-len(temp), stop=window)

        # *** scaling ***
        # price scale
        price_cols = [prefix+'open', prefix+'high', prefix+'low', prefix+'close']
        # add all columns wich need scale for one scale range with price
        for col in _columns:
            if 'ma' in col or 'bb_' in col or 'macd' in col:
                price_cols.append(col)
        # print(price_cols)
        temp[price_cols] = minmax_scale_array_global(temp[price_cols])

        # scaling not price-scale columns
        for col in _columns:
            if col not in price_cols:
                temp[[col]] = minmax_scale_array_global(temp[[col]])

        selected_data.append(temp)

        # divider
        if divider:
            try:
                temp['divider'] = np.zeros((window,)) 
            except Exception as ex:
                print(temp)
                raise ex
    
    # Конкатенация выбранных данных в один датафрейм
    result = pd.concat(selected_data, axis=1)
    
    # Заполнение значений NaN последним не NaN значением из соответствующего столбца
    result = result.fillna(method='ffill').fillna(method='bfill')
    
    return result