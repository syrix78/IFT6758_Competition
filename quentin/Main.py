import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webcolors

# In[1]:
if __name__ == "__main__":
    print("test")
    df_train = pd.read_csv("/Users/quentinwolak/Desktop/Cours/UdeM/quatrieme_annee/IFT6758 /Kaggle/ift6758-a20.nosync/lucas/IFT6758_Competition-lucas/train_cleaned.csv")

# In[2]:
df_train.info()

# In[2]:
likes = df_train["Num of Profile Likes"]
likes

# In[2]:
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# In[2]:
mask = ((df_train > (Q1 - 1.5 * IQR)) & (df_train < (Q3 + 1.5 * IQR)))
tmp = (df_train[mask["Avg Daily Profile Clicks"]])
mask.value_counts()

# In[]
df_train.groupby("Profile Page Color").value_counts()

# In[]
def delete_outliers(df, name_of_column):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ((df_train > (Q1 - 1.5 * IQR)) & (df_train < (Q3 + 1.5 * IQR)))
    return df[mask[name_of_column]]

# In[]
tmp = delete_outliers(df_train, "Num of Profile Likes")



# In[2]:
df_train.head()

# In[3]
df_train.isnull().sum()

# In[]
sns.countplot(x='Location', label="User Language", data=df_train);

# In[]
webcolors.hex_to_name(u'#daa520')