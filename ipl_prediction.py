import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

dfprev   = pd.read_csv("D:/sofware/Matches_IPL_2008-2019.csv")
df2020   = pd.read_csv('D:/sofware/Matches_IPL_2020.csv')
dftrain  = pd.read_csv("D:/sofware/Training.csv")
dfsampsub = pd.read_csv("D:/sofware/sample_submission.csv")

dftrain.head()

dfprev.info()

df2020.info()

df2020squads = pd.read_csv("D:/sofware/IPL_2020_Squads.csv",encoding= 'unicode_escape')
df2020squads

dftrain.shape
dftrain

dftrain['players'] = dftrain['Id']
dftrain['number'] =  dftrain['Id']
for i in range(0, len( dftrain)):
    dftrain['players'][i] =  dftrain['Id'][i].split("_")[-1]
    dftrain['number'][i] = int( dftrain['Id'][i].split('_')[:1][0])
    
dftrain.number = dftrain.number.astype(int)
dftrain.head()

plt.figure(figsize=(14,8),dpi=200)
sns.heatmap(dftrain.corr(),annot =True)

df = pd.DataFrame()
df['match_number'] = dftrain['number']
df['players'] = dftrain['players']
df['total_score'] = dftrain['Total Points']
df['Id']    = dftrain['Id']
df.head()

df.groupby(['players','match_number']).sum()

df['players'].value_counts()
df = df.drop(['players'],axis = 1)

df = df.iloc[:1283]
df.shape

df.match_number=  df.match_number.astype(int)
d =df.select_dtypes(include='object')
n = df.select_dtypes(exclude='object')
d.columns

d = pd.get_dummies(d,drop_first=True)
df = pd.concat([d,n],axis = 1)
df.head()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X= df.drop(['total_score'],axis=1)
y= df.total_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

y_pred = np.round(y_pred)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

dfsampsub['players'] = dfsampsub['Id']
dfsampsub['number'] =  dfsampsub['Id']
for i in range(0, len(dfsampsub)):
    dfsampsub['players'][i] =  dfsampsub['Id'][i].split("_")[-1]
    dfsampsub['number'][i] = int( dfsampsub['Id'][i].split('_')[:1][0])
dfsampsub

tp = dfsampsub.drop(['Total Points','players'],axis=1)
tp.number = tp.number.astype(int)
tp.number.dtypes
t = tp.select_dtypes(include='object') 
p = tp.select_dtypes(exclude='object')

t.columns
t= pd.get_dummies(t,drop_first=True)
t.columns

tp = pd.concat([t,p],axis=1)

tp.head()
scaler.fit(tp)
tp = scaler.transform(tp)
yp_pred = model.predict(tp)
yp_pred = np.round(yp_pred)

yp_pred  = [0 if i<0 else i for i in yp_pred]

dfsampsub['Total Points'] = yp_pred
dfsampsub = dfsampsub.drop(['players','number'],axis = 1)
dfsampsub.head()
dfsampsub.to_csv('IPL_Players_Performance_Prediction',index = False)

IPL_PLAYERS_PERFORMANCE_PREDICTION = pd.read_csv("./IPL_Players_Performance_Prediction")
IPL_PLAYERS_PERFORMANCE_PREDICTION.tail(20)
