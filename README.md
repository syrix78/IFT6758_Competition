# IFT6758_Competition: Another Team

Feature Engineering is GOOD!

## Links to Colab notebooks
* [data_cleaner.ipynb](https://drive.google.com/file/d/1gKCT40MCjgSa1LYEFach59ozHxF2J_T2/view?usp=sharing)
* [predictor.ipynb](https://drive.google.com/file/d/1GpX4-Rqc_R0r3F_BK6DOwSmQi0v1o9nn/view?usp=sharing)

## How to regenerate the `preds (11).csv` submission (Using Colab)

1. Delete `{train, test}_cleaned{.csv, .pkl}` files if they exist.
2. Upload the `train.csv` and `test.csv` files to the colab instance and run `data_cleaner.ipynb` to generate the cleaned data:
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
3. Download the files `{train, test}_cleaned.pkl` and upload them to another colab instance using the `predict.ipynb` file.
4. Run `predict.ipynb` to generate predictions using the instructions listed just below the title:
    1. Go the `Predict Kaggle Data` section
    2. Run all the cells in that section (Run the cells in order from top to bottom)
    3. You will see that a `predictions.csv` has been generated
    4. With `predictions.csv`, you should get the exact same predictions as in the `preds (11).csv` file we submitted on kaggle.

## How to approximately regenerate the `submission_5.csv` submission

1. Checkout the branch `submission_gradiant_boosting`.
2. Follow the instructions in its README file.

## How to approximately regenerate the `submission_8.csv` submission

1. Checkout the branch `submission_neural_net`.
2. Follow the instructions in its README file.