# IFT6758_Competition: Another Team

Feature Engineering is GOOD!

## How to regenerate the `submission_5.csv` submission

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
3. Run `predict.ipynb` to generate predictions.
    1. Go the `Predict Kaggle Data` section.
    2. Run all the cells in that section (Run the cells in order from top to bottom)
    3. You will see that a `predictions.csv` has been generated
    4. With `predictions.csv`, you should get the predictions that get a Kaggle score close to `submission_5.csv`.
