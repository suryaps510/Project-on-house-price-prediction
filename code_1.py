import numpy as np
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head(3)
test_df.head(3)
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(3)
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(3)
X_train = train_df.drop(['Id' ,'SalePrice','Alley','FireplaceQu','PoolQC','MiscFeature','Fence'] , axis=1 )
y_train = np.log(train_df['SalePrice'])        # Transform target variable with logarithm
X_test  = test_df.drop(['Id','Alley','FireplaceQu','PoolQC','MiscFeature','Fence'] , axis=1 )
import warnings
warnings.filterwarnings('ignore')
train_df['SalePrice'].describe()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.histplot(train_df['SalePrice'], kde=True , bins=30 , color= 'blue')
plt.title('Distribution of Sale Prices')
plt.show()
#Correlation heat map for numerical features
fig, ax = plt.subplots(figsize=(19, 9))
sns.heatmap(train_df.select_dtypes(include=['int64', 'float64']).corr(), annot=True , cmap='coolwarm' , fmt='.1f' , linewidth=.6)
plt.title('Correlation heatmap for numerical features')
plt.show()
k = 12
cols = train_df.select_dtypes(include=['int64', 'float64']).corr().nlargest(k, 'SalePrice')['SalePrice'].index  # nlargest pick the most powerfull correlation
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1)
sns.heatmap(cm, cbar=True, annot=True, cmap='coolwarm', square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF']
sns.pairplot(train_df[cols], height = 2.5 )
plt.show()
# This can show the relationship between the overall quality of a house and its sale price.
plt.figure(figsize=(15,8))
sns.boxplot(x='OverallQual' , y='SalePrice' , data=train_df)
plt.title('Sale Price box by Overall Quality')
data = pd.concat([train_df['SalePrice'] , train_df['GrLivArea'] ], axis=1)
data.plot.scatter(x='GrLivArea' , y='SalePrice')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64' , 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy='mean' )),  # filling nan or 0 with mean
    ('scaler' , StandardScaler() ) ])          # scalling the data with : z = (x - u) / s

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy='most_frequent' )),
    ('onehot' , OneHotEncoder(handle_unknown='ignore'))])


# preprocessing process
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=10)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])


# Train the model
my_pipeline.fit(X_train, y_train)
score = my_pipeline.score(X_train, y_train)
print(f"Model score: {score}") # model accuracy
# Predictions in log scale
predictions_log_scale = my_pipeline.predict(X_test)
# Convert predictions back from log scale
predictions = np.exp(predictions_log_scale)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions})



predictions_df.head()
predictions_df.to_csv("Final submission.csv", index=False)

