# IFT6758_Competition

Let's have fun!

## How to regenerate the `submission_8.csv` submission

1. Delete `{train, test}_cleaned{.csv, .pkl}` files if they exist.
2. Run `data_cleaner.ipynb` to generate the cleaned data.
    1. In the second code cell, set the first two lines as 
        ```
        generateTrain = True
        generateTest = False
        ```
    2. Run all the cells.
    3. In the second code cell, set the first two lines as 
        ```
        generateTrain = False
        generateTest = True
        ```
    4. Run all the cells.
3. Delete or move elsewhere hdf5 files generated in previous executions and that are present in the project's root folder.
4. Run `predict.ipynb` to generate predictions.
    1. Run all cells.
    2. To make predictions with the network that was just trained, in the 25th code cell load the weights from the hdf5 file with the lowest value after the "`--`" (it should be around `0.16`).
    3. Rerun the 2 last cells.
    4. With `submission_material/Weights-049--0.16242.hdf5` weights, you should get the exact same predictions as in the `submission_8.csv` file.