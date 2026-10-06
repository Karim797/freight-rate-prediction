from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
TARGET='posted_rate'
FEATURES=['pickup','delivery','route','pickup_lat','pickup_lon','delivery_lat','delivery_lon','distance','equipment','weight','market_index','quote_signal','month','day','dow','dayofyear']
CATS=['pickup','delivery','route','equipment']

def engineer(df):
    x=df.copy(); x['date']=pd.to_datetime(x['date'])
    x['route']=x['pickup'].astype(str)+'__'+x['delivery'].astype(str)
    x['month']=x.date.dt.month; x['day']=x.date.dt.day; x['dow']=x.date.dt.dayofweek; x['dayofyear']=x.date.dt.dayofyear
    return x

def fit_context(dev):
    ctx={}
    for city_col,lat,lon in [('pickup','pickup_lat','pickup_lon'),('delivery','delivery_lat','delivery_lon')]:
        g=dev.groupby(city_col)[[lat,lon]].median(); ctx[city_col]=g
    ctx['market_by_dow']=dev.groupby(dev.date.dt.dayofweek)['market_index'].median()
    ctx['market_median']=dev.market_index.median()
    ctx['quote_route']=dev.groupby(['pickup','delivery','equipment'])['quote_signal'].median()
    ctx['quote_median']=dev.quote_signal.median()
    return ctx

def add_context(x,ctx):
    x=x.copy()
    for city_col,lat,lon in [('pickup','pickup_lat','pickup_lon'),('delivery','delivery_lat','delivery_lon')]:
        if lat not in x: x[lat]=x[city_col].map(ctx[city_col][lat])
        if lon not in x: x[lon]=x[city_col].map(ctx[city_col][lon])
    if 'market_index' not in x: x['market_index']=x.date.dt.dayofweek.map(ctx['market_by_dow']).fillna(ctx['market_median'])
    if 'quote_signal' not in x:
        idx=pd.MultiIndex.from_frame(x[['pickup','delivery','equipment']]); x['quote_signal']=ctx['quote_route'].reindex(idx).to_numpy(); x['quote_signal']=x.quote_signal.fillna(ctx['quote_median'])
    return x

def prep(x, medians):
    z=x[FEATURES].copy()
    for c in CATS: z[c]=z[c].fillna('missing').astype(str)
    for c in [f for f in FEATURES if f not in CATS]: z[c]=pd.to_numeric(z[c],errors='coerce').fillna(medians[c])
    return z

def model(iterations=850):
    return CatBoostRegressor(iterations=iterations,depth=8,learning_rate=.07,loss_function='RMSE',l2_leaf_reg=6,random_seed=42,verbose=False,allow_writing_files=False)

dev=engineer(pd.read_csv(DATA/'train-test.csv')); ctx=fit_context(dev)
train=dev[dev.date<'2025-10-01'].copy(); hold=dev[dev.date>='2025-10-01'].copy()
med=train[[f for f in FEATURES if f not in CATS]].median(numeric_only=True)
m=model(200); m.fit(prep(train,med),train[TARGET],cat_features=[FEATURES.index(c) for c in CATS]); hp=m.predict(prep(hold,med))
metrics={'split':'Train Jan-Sep 2025; holdout Oct 2025','train_rows':len(train),'holdout_rows':len(hold),'MAE':float(mean_absolute_error(hold[TARGET],hp)),'RMSE':float(mean_squared_error(hold[TARGET],hp)**.5),'R2':float(r2_score(hold[TARGET],hp)),'MAPE_pct':float(np.mean(np.abs((hold[TARGET]-hp)/hold[TARGET]))*100)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
med_all=dev[[f for f in FEATURES if f not in CATS]].median(numeric_only=True); final=model(250); final.fit(prep(dev,med_all),dev[TARGET],cat_features=[FEATURES.index(c) for c in CATS])
val=add_context(engineer(pd.read_csv(DATA/'validation.csv')),ctx); pred=np.maximum(final.predict(prep(val,med_all)),1.0)
template=pd.read_csv(DATA/'validation-predictions-template.csv'); template['predicted_rate']=pd.Series(pred,index=val['load_id']).reindex(template['load_id']).to_numpy(); template.to_csv(ROOT/'validation_predictions.csv',index=False)
dec=add_context(engineer(pd.read_csv(DATA/'december-chart-inputs.csv')),ctx); dec['predicted_rate']=np.maximum(final.predict(prep(dec,med_all)),1.0)
dec[['pickup','delivery','distance','equipment','weight','date','predicted_rate']].assign(date=lambda d:d.date.dt.strftime('%Y-%m-%d')).to_csv(DATA/'december_chart_inputs.csv',index=False)
pd.DataFrame({'feature':FEATURES,'importance':final.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
print(json.dumps(metrics,indent=2)); print('Wrote validation_predictions.csv and data/december_chart_inputs.csv')
