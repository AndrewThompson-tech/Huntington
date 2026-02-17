import pandas as pd
import statistics

def chunkify(master_df: pd.DataFrame, yearly_periods: int) -> list:
    num_of_rows = yearly_periods * 12 # Get number of months
    length_of_df = len(master_df)
    all_chunks = []

    for i in range(0, length_of_df, 12): # Jump 1 year after each chunk
        start_index = i
        end_index = i + num_of_rows

        # grab a chunk
        chunk = master_df.iloc[start_index:end_index]

        # Only want to include equal length chunks
        # Ex: we use 3 years as our window, 25 / 3 will have a remainder which we don't want
        if len(chunk) == num_of_rows:
            all_chunks.append(chunk)

    return all_chunks

def compute_lagged_correlations(chunked_df: list[pd.DataFrame], macro_columns: list, etf_columns: list, num_of_lags: int) -> dict:
    start_lag = -num_of_lags # (-12) months
    end_lag = num_of_lags + 1 # exclusive 

    all_window_lags = {
        etf: {macro: [] for macro in macro_columns} 
        for etf in etf_columns
    }

    for window in chunked_df: 
        temp_etf_df = window[etf_columns]
        temp_macro_df = window[macro_columns]
        best_corr_matrix = None 
        best_lag_matrix = pd.DataFrame(index=etf_columns, columns=macro_columns)

        # utilizes pandas insanely optimal vectorizations; thanks to some cool C implementation
        for lag in range(start_lag, end_lag):  
            shifted_macro_df = temp_macro_df.shift(lag)
            # curr_corr_matrix = shifted_macro_df.corr(temp_etf_df)
            combined = pd.concat([shifted_macro_df, temp_etf_df], axis=1)
            corr_matrix = combined.corr()
            curr_corr_matrix = corr_matrix.loc[macro_columns, etf_columns]


            if best_corr_matrix is None: # handles the first iteration for each window
                best_corr_matrix =  curr_corr_matrix.copy() # we want a deep copy 
                best_lag_matrix[:] = lag # fill the entire matrix with this lag for now 

            mask = abs(curr_corr_matrix) > abs(best_corr_matrix) # boolean df that'll tell us which correlations are greater
            best_corr_matrix[mask] = curr_corr_matrix[mask] # will only change cells that are true from the mask
            best_lag_matrix[mask] = lag # for each true cell from the mask we update that cell to the new lag
        
        # append the results into our output 
        for etf in etf_columns:
            for macro in macro_columns:
                all_window_lags[etf][macro].append(best_lag_matrix.loc[etf, macro])
    
    return all_window_lags

def aggregate_lags(lagged_correlations: dict) -> dict:
    for etf in lagged_correlations:
        for macro in lagged_correlations[etf]:
            list_of_lags = lagged_correlations[etf][macro]
            optimal_lag = int(statistics.mode(list_of_lags)) 
            lagged_correlations[etf][macro] = optimal_lag

    return lagged_correlations