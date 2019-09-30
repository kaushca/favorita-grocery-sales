from datetime import date, timedelta
import datetime
import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

df_train = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "items.csv",
).set_index("item_nbr")

stores = pd.read_csv("stores.csv")

le=preprocessing.LabelEncoder()

items['family'] = le.fit_transform(items['family'].values)
items['class']=le.fit_transform(items['class'].values)
stores['type'] = le.fit_transform(stores['type'].values)
stores['state']=le.fit_transform(stores['state'].values)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods,freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods,freq=freq)]

def add_TS_features(df_2017,t2017, is_train=True):
    X = pd.DataFrame({
        "store_nbr":df_2017.index.get_level_values(0),
        "item_nbr":df_2017.index.get_level_values(1),
        "mean_1_2017": get_timespan(df_2017, t2017, 1, 1).mean(axis=1).values,
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "min_7_2017": get_timespan(df_2017,t2017,7,7).min(axis=1).values,
        "max_7_2017": get_timespan(df_2017,t2017,7,7).max(axis=1).values,
        "min_30_2017": get_timespan(df_2017,t2017,30,30).min(axis=1).values,
        "max_30_2017": get_timespan(df_2017,t2017,30,30).max(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values
    })
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

noOfWeeks=4
print("Preparing dataset...")
t2017 = date(2017, 6, 21)
X_l, y_l = [], []
for i in range(noOfWeeks):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = add_TS_features(df_2017,t2017 + delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = add_TS_features(df_2017,date(2017, 7, 26))
X_test = add_TS_features(df_2017,date(2017, 8, 16), is_train=False)

X_train["family"]=pd.concat([items['family']] * noOfWeeks).values
X_val['family'] = items['family'].values
X_test['family'] = items['family'].values
X_train['class'] = pd.concat([items['class']] * noOfWeeks).values
X_val['class'] = items['class'].values
X_test['class'] = items['class'].values

X_train = pd.merge(X_train, stores, on='store_nbr', how='left')
X_val = pd.merge(X_val, stores, on='store_nbr', how='left')
X_test = pd.merge(X_test, stores, on='store_nbr', how='left')

del(X_train["city"])
del(X_val["city"])
del(X_test["city"])

print("Training and predicting models...")
xgb=XGBRegressor(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=1,
    objective='reg:linear',
    nthread=4,
    seed=13456,
    eval_metric='rmse'    
)

val_pred=[]
test_pred=[]
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    eval_set=[(X_val,y_val[:,i])]
    bst=xgb.fit(X_train,y_train[:,i],sample_weight=pd.concat([items["perishable"]] * noOfWeeks) * 0.25 + 1,
        eval_set=eval_set,verbose=True,early_stopping_rounds=50
    )
    val_pred.append(bst.predict(X_val))
    test_pred.append(bst.predict(X_test))

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar",show=False)
plt.savefig("summary_plot_xgb.png",bbox_inches="tight")

print("Validation mse:", mean_squared_error(y_val, np.array(val_pred).transpose()))

weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)
print('nwrmsle = {}'.format(err))

def mape(a, b):
    b=b.transpose()
    e = 2 * abs(a - b) / ( abs(a)+abs(b) )
    return (e.mean())

mape_value=mape(np.expm1(y_val),np.expm1(np.array(val_pred)))
print("MAPE = {}".format(mape_value))
print("Making submission...")

y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('xgb.csv', float_format='%.4f', index=None)
