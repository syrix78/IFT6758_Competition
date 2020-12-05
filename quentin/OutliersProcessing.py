
"""
This function will take two parameters that are:
df = a dataframe
name_of_column = the name of the column to process
and will return the dataframe without the line where there were outliers in the column
The work is done by using the IQR outliers
"""
def delete_outliers(df, name_of_column):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ((df > (Q1 - 1.5 * IQR)) & (df < (Q3 + 1.5 * IQR)))
    return df[mask[name_of_column]]