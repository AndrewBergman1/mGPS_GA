import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def load_vif_file() : 
    vif_df = pd.read_csv("vif_df.csv")
    return vif_df

def is_row_valid(row):
    """
    Returns True if the row's VIF value (assumed to be in the third column) is less than 5.
    This function is designed to be used with individual rows of a DataFrame.
    """
    return row[1] < 5

def filter_vifs(vif_df):
    """
    Filters the DataFrame to keep only the rows where the VIF value is less than 5.
    """
    # Convert DataFrame to a list of tuples (rows) to be processed in parallel
    rows = [tuple(x) for x in vif_df.to_numpy()]
    
    # Use a ProcessPoolExecutor to parallelize the operation
    with ProcessPoolExecutor() as executor:
        # The map function applies is_row_valid to each row
        results = list(executor.map(is_row_valid, rows))
    
    # Use the results to filter the DataFrame
    # Convert results to a boolean series and filter the original DataFrame
    filtered_vifs = vif_df[results]
    
    return filtered_vifs

def check_column(column_name, df_columns):
    """
    Returns the column name if it should be removed, otherwise returns None.
    """
    if column_name not in df_columns:
        return column_name
    return None

def filter_predictors(filtered_vifs):
    abundance_file = pd.read_csv("metasub_taxa_abundance.csv")
    df_columns = set(abundance_file.columns)

    # Extract column names from the filtered_vifs tuples (assuming the column name is the first element)
    column_names = [element[0] for element in filtered_vifs]

    # Initialize ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        # Map check_column function across all elements, passing the set of DataFrame column names for comparison
        columns_to_remove = filter(None, executor.map(check_column, column_names, [df_columns] * len(column_names)))

    # Remove columns that were marked for removal
    abundance_file.drop(columns=columns_to_remove, errors='ignore', inplace=True)

    abundance_file.to_csv("Filtered_predictors_on_vif")

vif_df = load_vif_file()
filtered_vifs = filter_vifs(vif_df)
filter_predictors(filtered_vifs)


