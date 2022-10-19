from statsmodels.formula.api import ols
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr
from scipy.stats import ttest_rel
from numpy import sqrt, abs, round
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest as ztest
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr
from scipy.stats import ttest_rel
from numpy import sqrt, abs, round
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest as ztest
# machine learning
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV

import lightgbm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import global_config as gcf

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from statsmodels.stats.outliers_influence import OLSInfluence as olsi


from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima


# Load the data frame
df_orginal = pd.read_csv("Facility/TRI_By_ID_1987_2014.csv")

# df_orginal = df_orginal.iloc[:1000, :]
# Summary Statistics
print(df_orginal.head())
print(df_orginal.info())
print(df_orginal.describe())

# Check the missing values
print(df_orginal.isnull().values.any())  # tells nan are present or not
print(df_orginal.isnull().any())  # shows column wise nans occurance
print(df_orginal.isnull().sum())  # gives total count of nans column wise

df_chemicals = df_orginal.drop(['tri_facility_id', 'year', 'facility_name', 'street_address', 'city',
                                'county', 'st', 'zip', 'federal_facility', 'parent_company_name'], axis=1)
print(df_chemicals.head())
# imputation
for column in df_chemicals.columns.tolist():
    if df_chemicals[column].isnull().any():
        df_chemicals[column] = df_chemicals[column].fillna(
            df_chemicals[column].mode()[0])
print(df_chemicals.isnull().any())

# exploratory analysis
# year
df_year = df_chemicals[['total_release_carcinogen',
                        'total_release_metal']]
df_year['year'] = df_orginal['year']
df = df_year.groupby('year').sum()
df['year'] = df.index
fig, ax = plt.subplots()
df.plot.bar(x='year', y=['total_release_carcinogen',
            'total_release_metal'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Year", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Yearly - Carcinogen & Metal Release", size=18)
plt.savefig("cm_bar.png")  # save image
plt.show()

# federal_facility
df_year = df_chemicals[['total_release_carcinogen',
                        'total_release_metal']]
df_year['federal_facility'] = df_orginal['federal_facility']
df = df_year.groupby('federal_facility').sum()
df['federal_facility'] = df.index
fig, ax = plt.subplots()
df.plot.bar(x='federal_facility', y=['total_release_carcinogen',
            'total_release_metal'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Federal Facility", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Federal Facility - Carcinogen & Metal Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()


# county
df_county = df_chemicals[['total_release_carcinogen']]
df_county['county'] = df_orginal['county']
df = df_county.groupby('county').sum()
fig, ax = plt.subplots()
df_sorted = df.sort_values(by=['total_release_carcinogen'],
                           ascending=False)
df_merged = pd.concat([df_sorted.head(10), df_sorted.tail(10)],
                      axis=1)
df_merged['county'] = df_merged.index
df_merged.plot.bar(x='county', y=['total_release_carcinogen'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("County", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Top and Bottom 10 County Carcinogen Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

df_county = df_chemicals[['total_release_metal']]
df_county['county'] = df_orginal['county']
df = df_county.groupby('county').sum()
fig, ax = plt.subplots()
df_sorted = df.sort_values(by=['total_release_metal'],
                           ascending=False)
df_merged = pd.concat([df_sorted.head(10), df_sorted.tail(10)],
                      axis=1)
df_merged['county'] = df_merged.index
df_merged.plot.bar(x='county', y=['total_release_metal'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("County", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Top and Bottom 10 County Metal Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

# parent_company_name
df_parent_company_name = df_chemicals[['total_release_carcinogen']]
df_parent_company_name['parent_company_name'] = df_orginal['parent_company_name']
df = df_parent_company_name.groupby('parent_company_name').sum()
fig, ax = plt.subplots()
df_sorted = df.sort_values(by=['total_release_carcinogen'],
                           ascending=False)
df_merged = pd.concat([df_sorted.head(10), df_sorted.tail(10)],
                      axis=1)
df_merged['parent_company_name'] = df_merged.index
df_merged.plot.bar(x='parent_company_name', y=[
                   'total_release_carcinogen'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Parent Company", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Top and Bottom 10 Parent Company Carcinogen Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

df_parent_company_name = df_chemicals[['total_release_metal']]
df_parent_company_name['parent_company_name'] = df_orginal['parent_company_name']
df = df_parent_company_name.groupby('parent_company_name').sum()
fig, ax = plt.subplots()
df_sorted = df.sort_values(by=['total_release_metal'],
                           ascending=False)
df_merged = pd.concat([df_sorted.head(10), df_sorted.tail(10)],
                      axis=1)
df_merged['parent_company_name'] = df_merged.index
df_merged.plot.bar(x='parent_company_name', y=[
                   'total_release_metal'], rot=40, ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Parent Company", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Top and Bottom 10 Parent Company Metal Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

# state
df_st = df_chemicals[['total_release_carcinogen']]
df_st['st'] = df_orginal['st']
df = df_st.groupby('st').sum()
df = df.sort_values(by=['total_release_carcinogen'],
                    ascending=False).iloc[:25, :]
plot = df.plot.pie(y='total_release_carcinogen', figsize=(
    11, 6), title='Top 25 Carcinogen Release States', autopct='%1.0f%%')
plt.show()

df_st = df_chemicals[['total_release_metal']]
df_st['st'] = df_orginal['st']
df = df_st.groupby('st').sum()
df = df.sort_values(by=['total_release_metal'],
                    ascending=False).iloc[:25, :]
plot = df.plot.pie(y='total_release_metal', figsize=(
    11, 6), title='Top 25 Metal Release States', autopct='%1.0f%%')
plt.show()

# city
df_city = df_chemicals[['total_release_carcinogen']]
df_city['city'] = df_orginal['city']
df = df_city.groupby('city').sum()
df = df.sort_values(by=['total_release_carcinogen'],
                    ascending=False).iloc[:25, :]
plot = df.plot.pie(y='total_release_carcinogen', figsize=(
    11, 6), title='Top 25 Carcinogen Release Cities', autopct='%1.0f%%')
plt.show()

df_city = df_chemicals[['total_release_metal']]
df_city['city'] = df_orginal['city']
df = df_city.groupby('city').sum()
df = df.sort_values(by=['total_release_metal'],
                    ascending=False).iloc[:25, :]
plot = df.plot.pie(y='total_release_metal', figsize=(
    11, 6), title='Top 25 Metal Release Cities', autopct='%1.0f%%')
plt.show()

# correlation analysis
df = df_chemicals
df['year'] = df_orginal['year']
df['federal_facility'] = [
    1 if i == 'YES' else 0 for i in df_orginal['federal_facility']]
cormat = df.corr()
print(round(cormat, 2))

f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.title("Heatmap: Toxic Release Data", size=18)
plt.savefig("guns_hm.png")  # save image
plt.show()

# hypothesis testing
def two_sample_t_test(sample1, sample2, alpha):
    """
    Conducts t-test on independant features at a given alpha
    :param sample1: name of feature1
    :param sample2: name of feature2
    """
    # Compute the descriptive statistics of a and b.
    sample1_mean = sample1.mean()
    sample1_var = sample1.var(ddof=1)
    sample1_sd = sample1.std()
    sample1_n = sample1.size
    sample1_dof = sample1_n - 1

    sample2_mean = sample2.mean()
    sample2_var = sample2.var(ddof=1)
    sample2_sd = sample2.std()
    sample2_n = sample2.size
    sample2_dof = sample2_n - 1

    # Use scipy.stats.ttest_ind_from_stats.
    t_val, p_val = ttest_ind_from_stats(sample1_mean, np.sqrt(sample1_var), sample1_n,
                                        sample2_mean, np.sqrt(
                                            sample2_var), sample2_n,
                                        equal_var=False)
    t_val = np.round(t_val, 8)
    p_val = np.round(p_val, 6)
    if (p_val < alpha):
        Hypothesis_Status = 'Status: Reject Null Hypothesis'
    else:
        Hypothesis_Status = 'Status: Do Not Reject Null Hypothesis'
    return t_val, p_val, Hypothesis_Status


def two_sample_z_test(sample1, sample2, alpha):
    """
    Conducts z-test on independant features at a given alpha
    :param sample1: name of feature1
    :param sample2: name of feature2
    """
    m1, m2 = sample1.mean(), sample2.mean()
    sd1, sd2 = sample1.std(), sample2.std()
    n1, n2 = sample1.shape[0], sample2.shape[0]
    ovr_sigma = sqrt(sd1**2/n1 + sd2**2/n2)
    z = (m1 - m2)/ovr_sigma
    p = 2*(1 - norm.cdf(abs(z)))
    z_score = np.round(z, 8)
    p_val = np.round(p, 6)
    if (p_val < alpha):
        Hypothesis_Status = 'Status: Reject Null Hypothesis'
    else:
        Hypothesis_Status = 'Status: Do Not Reject Null Hypothesis'
    return z_score, p_val, Hypothesis_Status


# federal_facility
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['parent_company_name'] = df_orginal['parent_company_name']
df = df.groupby('parent_company_name').sum()
df['parent_company_name'] = df.index

sample1 = df['total_release_carcinogen'].to_numpy()
sample2 = df['total_release_metal'].to_numpy()

print('parental company--------->>>>')
t_score, p_val, Hypothesis_Status = two_sample_t_test(sample1, sample2, 0.05)
print('T Score: ', t_score, p_val, Hypothesis_Status)
z_score, p_val, Hypothesis_Status = two_sample_z_test(sample1, sample2, 0.05)
print('Z Score: ', t_score, p_val, Hypothesis_Status)


# county
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['county'] = df_orginal['county']
df = df.groupby('county').sum()
df['county'] = df.index

sample1 = df['total_release_carcinogen'].to_numpy()
sample2 = df['total_release_metal'].to_numpy()

print('county--------->>>>')
t_score, p_val, Hypothesis_Status = two_sample_t_test(sample1, sample2, 0.05)
print('T Score: ', t_score, p_val, Hypothesis_Status)
z_score, p_val, Hypothesis_Status = two_sample_z_test(sample1, sample2, 0.05)
print('Z Score: ', t_score, p_val, Hypothesis_Status)

# state
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['st'] = df_orginal['st']
df = df.groupby('st').sum()
df['st'] = df.index

sample1 = df['total_release_carcinogen'].to_numpy()
sample2 = df['total_release_metal'].to_numpy()

print('state--------->>>>')
t_score, p_val, Hypothesis_Status = two_sample_t_test(sample1, sample2, 0.05)
print('T Score: ', t_score, p_val, Hypothesis_Status)
z_score, p_val, Hypothesis_Status = two_sample_z_test(sample1, sample2, 0.05)
print('Z Score: ', t_score, p_val, Hypothesis_Status)

# city
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['city'] = df_orginal['city']
df = df.groupby('city').sum()
df['city'] = df.index

sample1 = df['total_release_carcinogen'].to_numpy()
sample2 = df['total_release_metal'].to_numpy()

print('city--------->>>>')
t_score, p_val, Hypothesis_Status = two_sample_t_test(sample1, sample2, 0.05)
print('T Score: ', t_score, p_val, Hypothesis_Status)
z_score, p_val, Hypothesis_Status = two_sample_z_test(sample1, sample2, 0.05)
print('Z Score: ', t_score, p_val, Hypothesis_Status)


df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['year'] = df_orginal['year']
df['federal_facility'] = df_orginal['federal_facility']

#df = df.groupby('year')
fig, ax = plt.subplots()
df.boxplot('total_release_carcinogen', by='year', color='red', ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Year", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Yearly Carcinogen Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

fig, ax = plt.subplots()
df.boxplot('total_release_metal', by='year', color='red', ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Year", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Yearly Metal Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

sns.violinplot(x='year', y='total_release_carcinogen', data=df)
plt.xlabel("Year", size=18)
plt.ylabel("Total Release", size=18)
plt.title("Violinplot: Yearly Carcinogen Release", size=24)
plt.show()

sns.violinplot(x='year', y='total_release_metal', data=df)
plt.xlabel("Year", size=18)
plt.ylabel("Total Release", size=18)
plt.title("Violinplot: Yearly Metal Release", size=24)
plt.show()

fig, ax = plt.subplots()
df.boxplot('total_release_carcinogen',
           by='federal_facility', color='red', ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Federal Facility", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Federal Facility Carcinogen Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()


fig, ax = plt.subplots()
df.boxplot('total_release_metal', by='federal_facility', color='red', ax=ax)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()))
plt.xlabel("Federal Facility", size=14)
plt.ylabel("Total Release", size=14)
plt.title("Federal Facility Metal Release", size=18)
plt.savefig("ff_bar.png")  # save image
plt.show()

sns.violinplot(x='federal_facility', y='total_release_carcinogen', data=df)
plt.xlabel("Federal Facility", size=18)
plt.ylabel("Total Release", size=18)
plt.title("Violinplot: Federal Facility Carcinogen Release", size=24)
plt.show()

sns.violinplot(x='federal_facility', y='total_release_metal', data=df)
plt.xlabel("Federal Facility", size=18)
plt.ylabel("Total Release", size=18)
plt.title("Violinplot: Federal Facility Metal Release", size=24)
plt.show()


# anova
print(" year anova-----------")
mod = ols('total_release_carcinogen ~ year',
          data=df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)
print('total_release_carcinogen', aov_table)

mod = ols('total_release_metal ~ year',
          data=df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)
print('total_release_metal', aov_table)

print(" federal facility anova-----------")
mod = ols('total_release_carcinogen ~ federal_facility',
          data=df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)
print('total_release_carcinogen', aov_table)

mod = ols('total_release_metal ~ federal_facility',
          data=df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)
print('total_release_metal', aov_table)

# two way anova
print('two way anova total_release_carcinogen---------')
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['year'] = df_orginal['year']
df['federal_facility'] = df_orginal['federal_facility']
model = ols(
    'total_release_carcinogen ~ C(year) + C(federal_facility) + C(year):C(federal_facility)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)

print(aov_table.round(4))

print('total_release_metal---------')
df = df_chemicals[['total_release_carcinogen',
                   'total_release_metal']]
df['year'] = df_orginal['year']
df['federal_facility'] = df_orginal['federal_facility']
model = ols(
    'total_release_metal ~ C(year) + C(federal_facility) + C(year):C(federal_facility)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)

print(aov_table.round(4))

# train test split
def train_test_break(df, target_var, train_size=0.7):
    """
    Creates train and test datasets
    :param df: dataframe of loaded data
    :param target_var: name of target variable
    :return: train and test datasets
    """
    train, test = train_test_split(
        df, train_size=train_size, stratify=df[target_var], random_state=42)
    Y_train = train[target_var]
    Y_test = test[target_var]
    X_train = train.drop([target_var], axis=1)
    X_test = test.drop([target_var], axis=1)
    return(X_train, X_test, Y_train, Y_test)


# training linear regression for carcinogen prediction
def linear_regression(X_train, X_test, y_train, y_test, title):
    model = LinearRegression()
    model.fit(X_train,y_train)
    print(model.intercept_)
    coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
    print(coeff_parameter)

    predictions = model.predict(X_test)
    sns.regplot(y_test,predictions)
    plt.title(title, size=18)
    plt.ylabel("Toxic Release", size=14)
    plt.show()

    y_pred = model.predict(X_test)
    #print(y_pred)
    linear_rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
    X_train_Sm= sm.add_constant(X_train)
    X_train_Sm= sm.add_constant(X_train)
    ls=sm.OLS(y_train,X_train_Sm).fit()
    print(ls.summary())
    return ls, linear_rmse


#total carcinogen release
df = df_chemicals
df['year'] = df_orginal['year']
X = df.drop(['total_release_carcinogen'],axis=1)
Y = df['total_release_carcinogen']


#stepwise regression
lreg = LinearRegression()
sfs_ = sfs(lreg, k_features=10, forward=True, verbose=2, scoring='r2', cv=5)
sfs_ = sfs_.fit(X, Y)
feat_names = list(sfs_.k_feature_names_)
print(feat_names)

#feat_names = ['ammonia', 'chromium', 'lead', 'methanol', 'styrene']
#model
#regression model
X = df[feat_names]
Y = df['total_release_carcinogen']
#data splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

lm, linear_rmse = linear_regression(X_train, X_test, y_train, y_test, 'Linear Regression Total Carcinogen Release')


## lasso regression
# generate lambda values
lambda_values1 = np.random.randint(0,1000,100)
lambda_values_2 = np.linspace(0.01,500,100)
alphas = lambda_values1 + lambda_values_2
alphas = 10**np.linspace(10,-2,100)*0.5 #
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');
plt.show()

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1, max_iter=100)
# fit model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(y_pred)
lasso_rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
print(lasso_rmse)
print("best alpha--",model.alpha_)


#best model
# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

y_pred = lasso_best.predict(X_test)
#print(y_pred)
best_rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
print(linear_rmse, " | ", lasso_rmse, " | ", best_rmse, " | ", model.alpha_)

predictions = lasso_best.predict(X_test)
sns.regplot(y_test,predictions)
plt.title('Lasso Best Linear Regression Model', size=18)
plt.ylabel("Toxic Release", size=14)
plt.show()

##total metal release
df = df_chemicals
df['year'] = df_orginal['year']
X = df.drop(['total_release_metal'],axis=1)
Y = df['total_release_metal']

#stepwise regression
lreg = LinearRegression()
sfs_ = sfs(lreg, k_features=12, forward=True, verbose=2, scoring='r2', cv=5)
sfs_ = sfs_.fit(X, Y)
feat_names = list(sfs_.k_feature_names_)
print(feat_names)

#feat_names = ['ammonia', 'chromium', 'lead', 'methanol', 'styrene']
#model
#regression model
X = df[feat_names]
Y = df['total_release_metal']
#data splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

lm, linear_rmse = linear_regression(X_train, X_test, y_train, y_test, 'Linear Regression Total Metal Release')

## lasso regression
# generate lambda values
lambda_values1 = np.random.randint(0,1000,100)
lambda_values_2 = np.linspace(0.01,500,100)
alphas = lambda_values1 + lambda_values_2
alphas = 10**np.linspace(10,-2,100)*0.5 #
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha')
plt.show()

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1, max_iter=100)
# fit model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(y_pred)
lasso_rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
print(lasso_rmse)
print("best alpha--",model.alpha_)


#best model
# Set best alpha
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

y_pred = lasso_best.predict(X_test)
#print(y_pred)
best_rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
print(linear_rmse, " | ", lasso_rmse, " | ", best_rmse, " | ", model.alpha_)

predictions = lasso_best.predict(X_test)
sns.regplot(y_test,predictions)
plt.title('Lasso Best Linear Regression Model', size=18)
plt.ylabel("Toxic Release", size=14)
plt.show()


## use variables and fit a regression model
def logistic_regression(X_train, X_test, Y_train, Y_test,title):

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train,Y_train)

    predictions = model.predict(X_test)
    cm = metrics.confusion_matrix(Y_test, predictions)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix - {}'.format(title), y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    #roc curve
    preds_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_test,  preds_proba)
    auc = metrics.roc_auc_score(Y_test, preds_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title('ROC Curve - {}'.format(title))
    plt.show()

    print("Accuracy:",metrics.accuracy_score(Y_test, predictions))
    print("Precision:",metrics.precision_score(Y_test, predictions))
    print("Recall:",metrics.recall_score(Y_test, predictions))
    print("F1 Score:",metrics.f1_score(Y_test, predictions))

    return ''

# preprocess data
X = df
target_feature = df_orginal['federal_facility']
Y = [1 if i=='YES' else 0 for i in target_feature]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# simple model
results = logistic_regression(X_train, X_test, Y_train, Y_test,'Logistc Regression Model')
print(results)

def lgb(X_train, X_test, Y_train, Y_test):
    """
    Builds lgd model on train and test datasets
    :param X_train: train dataset
    :param X_test: test dataset
    :param Y_train: train target list
    :param Y_test: test target list
    :return: tuple of model metrics and trained model
    """
    parameters = gcf.lgb_params
    clf = lightgbm.LGBMClassifier()
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    accuracy = round(accuracy_score(Y_test, preds) * 100, 2)
    cm = confusion_matrix(Y_test, preds)
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')
    f1 = f1_score(Y_test, preds, average='weighted')
    print('-'*80)
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)
    recall = round(recall, 2)
    print()
    print('Light Gradient Boosting Validation Accuracy: ', accuracy)
    print('Confusion Matrix:')
    print(cm)
    print()

    return(clf, preds, accuracy, cm, precision, recall, f1)

def xgboost(X_train, X_test, Y_train, Y_test):
    """
    Builds xgboost model on train and test datasets
    :param X_train: train dataset
    :param X_test: test dataset
    :param Y_train: train target list
    :param Y_test: test target list
    :return: tuple of model metrics and trained model
    """
    parameters = gcf.xgb_params
    clf = xgb.XGBClassifier(**parameters)
    acc_scorer = make_scorer(accuracy_score)
    parameters['objective'] = 'binary:logistic'
    print("XGBoost Model is getting trained -------------->>>")
    clf.fit(X_train, Y_train)
    predictions = clf.predict_proba(X_test)
    preds = []
    for i in predictions:
        if i[1] > 0.4:
            preds.append(1)
        else:
            preds.append(0)
    accuracy = round(accuracy_score(Y_test, preds) * 100, 2)
    cm = confusion_matrix(Y_test, preds)
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')
    f1 = f1_score(Y_test, preds, average='weighted')
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)
    recall = round(recall, 2)
    print()
    print('Extreme Gradient Boosting Validation Accuracy: ', accuracy)
    print('Confusion Matrix:')
    print(cm)

    return(clf, preds, accuracy, cm, precision, recall, f1)


def random_forest(X_train, X_test, Y_train, Y_test):
    """
    Builds random forest model on train and test datasets
    :param X_train: train dataset
    :param X_test: test dataset
    :param Y_train: train target list
    :param Y_test: test target list
    :return: tuple of model metrics and trained model
    """
    parameters = gcf.rf_params
    clf = RandomForestClassifier(**parameters)
    acc_scorer = make_scorer(accuracy_score)
    print("Random Forest Model is getting trained -------------->>>")
    clf.fit(X_train, Y_train)
    predictions = clf.predict_proba(X_test)
    preds = []
    for i in predictions:
        if i[1] > 0.4:
            preds.append(1)
        else:
            preds.append(0)
    accuracy = round(accuracy_score(Y_test, preds) * 100, 2)
    cm = confusion_matrix(Y_test, preds)
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')
    f1 = f1_score(Y_test, preds, average='weighted')
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)
    recall = round(recall, 2)
    print()
    print('Random Forest Validation Accuracy: ', accuracy)
    print('Confusion Matrix:')
    print(cm)
    print()

    return(clf, preds, accuracy, cm, precision, recall, f1)


def svm(X_train, X_test, Y_train, Y_test):
    """
    Builds svm model on train and test datasets
    :param X_train: train dataset
    :param X_test: test dataset
    :param Y_train: train target list
    :param Y_test: test target list
    :return: tuple of model metrics and trained model
    """
    parameters = gcf.svc_params
    clf = SVC(**parameters)
    acc_scorer = make_scorer(accuracy_score)
    print("SVM Model is getting trained -------------->>>")
    clf.fit(X_train, Y_train)
    predictions = clf.predict_proba(X_test)
    preds = []
    for i in predictions:
        if i[1] > 0.4:
            preds.append(1)
        else:
            preds.append(0)
    accuracy = round(accuracy_score(Y_test, preds) * 100, 2)
    cm = confusion_matrix(Y_test, preds)
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')
    f1 = f1_score(Y_test, preds, average='weighted')
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)
    recall = round(recall, 2)
    print()
    print('SVM Validation Accuracy: ', accuracy)
    print('Confusion Matrix:')
    print(cm)
    print()

    return(clf, preds, accuracy, cm, precision, recall, f1)


def naive_bayes(X_train, X_test, Y_train, Y_test):
    """
    Builds naive bayes model on train and test datasets
    :param X_train: train dataset
    :param X_test: test dataset
    :param Y_train: train target list
    :param Y_test: test target list
    :return: tuple of model metrics and trained model
    """
    parameters = gcf.nb_params
    clf = GaussianNB(**parameters)
    print("Naive Bayes Model is getting trained -------------->>>")
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    accuracy = round(accuracy_score(Y_test, preds) * 100, 2)
    cm = confusion_matrix(Y_test, preds)
    precision = precision_score(Y_test, preds, average='weighted')
    recall = recall_score(Y_test, preds, average='weighted')
    f1 = f1_score(Y_test, preds, average='weighted')
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)
    recall = round(recall, 2)
    print()
    print('Gaussian Naive Bayes Validation Accuracy: ', accuracy)
    print('Confusion Matrix:')
    print(cm)
    print()

    return(clf, preds, accuracy, cm, precision, recall, f1)


#lgb
clf, preds, accuracy, cm, precision, recall, f1 = lgb(X_train, X_test, Y_train, Y_test)
print(accuracy, cm, precision, recall, f1)

#xgboost
clf, preds, accuracy, cm, precision, recall, f1 = xgboost(X_train, X_test, Y_train, Y_test)
print(accuracy, cm, precision, recall, f1)

#random_forest
clf, preds, accuracy, cm, precision, recall, f1 = random_forest(X_train, X_test, Y_train, Y_test)
print(accuracy, cm, precision, recall, f1)

#svm
clf, preds, accuracy, cm, precision, recall, f1 = svm(X_train, X_test, Y_train, Y_test)
print(accuracy, cm, precision, recall, f1)

#naive_bayes
clf, preds, accuracy, cm, precision, recall, f1 = naive_bayes(X_train, X_test, Y_train, Y_test)
print(accuracy, cm, precision, recall, f1)


#time series
df = df_chemicals
df['year'] = df_orginal['year']
df = df.groupby('year').sum()
#df['year'] = pd.to_datetime(df.index)

carcinogen = df[['total_release_carcinogen']]
metal = df[['total_release_metal']]
#carcinogen = carcinogen.set_index('year')
#metal = metal.set_index('year')

# the simple moving average over a period of 10 years
carcinogen['Simple Moving Average 3'] = carcinogen['total_release_carcinogen'].rolling(3, min_periods=1).mean()
carcinogen.plot()
plt.title("Total Carcinogen Release Observed Data", size=18)
plt.show()
# the simple moving average over a period of 10 years
metal['Simple Moving Average 3'] = metal['total_release_metal'].rolling(3, min_periods=1).mean()
metal.plot()
plt.title("Total Metal Release Observed Data", size=18)
plt.show()

carcinogen_ts = carcinogen.drop(['Simple Moving Average 3'], axis=1)
metal_ts = metal.drop(['Simple Moving Average 3'], axis=1)

fit1 = SimpleExpSmoothing(carcinogen_ts, initialization_method="heuristic").fit(
    smoothing_level=0.2, optimized=False
)
fcast1 = fit1.forecast(5).rename(r"$\alpha=0.2$")
fit2 = SimpleExpSmoothing(carcinogen_ts, initialization_method="heuristic").fit(
    smoothing_level=0.6, optimized=False
)
fcast2 = fit2.forecast(5).rename(r"$\alpha=0.6$")
fit3 = SimpleExpSmoothing(carcinogen_ts, initialization_method="estimated").fit()
fcast3 = fit3.forecast(5).rename(r"$\alpha=%s$" % fit3.model.params["smoothing_level"])

plt.figure(figsize=(12, 8))
plt.plot(carcinogen_ts, marker="o", color="black")
plt.plot(fit1.fittedvalues, marker="o", color="blue")
(line1,) = plt.plot(fcast1, marker="o", color="blue")
plt.plot(fit2.fittedvalues, marker="o", color="red")
(line2,) = plt.plot(fcast2, marker="o", color="red")
plt.plot(fit3.fittedvalues, marker="o", color="green")
(line3,) = plt.plot(fcast3, marker="o", color="green")
plt.plot(carcinogen['total_release_carcinogen'], marker="o", color="orange")
(line4,) = plt.plot(carcinogen['total_release_carcinogen'], marker="o", color="orange")
plt.plot(carcinogen['Simple Moving Average 3'], marker="o", color="grey")
(line5,) = plt.plot(carcinogen['Simple Moving Average 3'], marker="o", color="grey")
plt.legend([line1, line2, line3, line4, line5], [fcast1.name, fcast2.name, fcast3.name, 'total_release_carcinogen', 'Simple Moving Average 3'])
plt.title("Exponential Smoothing - Total Carcinogen Release", size=18)
plt.xlim([1985, 2020])
plt.show()


fit1 = SimpleExpSmoothing(metal_ts, initialization_method="heuristic").fit(
    smoothing_level=0.2, optimized=False
)
fcast1 = fit1.forecast(5).rename(r"$\alpha=0.2$")
fit2 = SimpleExpSmoothing(metal_ts, initialization_method="heuristic").fit(
    smoothing_level=0.6, optimized=False
)
fcast2 = fit2.forecast(5).rename(r"$\alpha=0.6$")
fit3 = SimpleExpSmoothing(metal_ts, initialization_method="estimated").fit()
fcast3 = fit3.forecast(5).rename(r"$\alpha=%s$" % fit3.model.params["smoothing_level"])

plt.figure(figsize=(12, 8))
plt.plot(metal_ts, marker="o", color="black")
plt.plot(fit1.fittedvalues, marker="o", color="blue")
(line1,) = plt.plot(fcast1, marker="o", color="blue")
plt.plot(fit2.fittedvalues, marker="o", color="red")
(line2,) = plt.plot(fcast2, marker="o", color="red")
plt.plot(fit3.fittedvalues, marker="o", color="green")
(line3,) = plt.plot(fcast3, marker="o", color="green")
plt.plot(metal['total_release_metal'], marker="o", color="orange")
(line4,) = plt.plot(metal['total_release_metal'], marker="o", color="orange")
plt.plot(metal['Simple Moving Average 3'], marker="o", color="grey")
(line5,) = plt.plot(metal['Simple Moving Average 3'], marker="o", color="grey")
plt.legend([line1, line2, line3, line4, line5], [fcast1.name, fcast2.name, fcast3.name, 'total_release_metal', 'Simple Moving Average 3'])
plt.title("Exponential Smoothing - Total Metal Release", size=18)
plt.xlim([1985, 2020])
plt.show()


if __name__ == '__main__':
    print("Group7 - Final Project")

