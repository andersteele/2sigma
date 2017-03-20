from sklearn import linear_model
import numpy as np 
import pandas as pd 

#Hyperparameters
M=1
alpha = 1.0 #Ridge
low_y = -0.075
high_y = 0.075
A = 0.2

#initializing & helper function
m1_feat=[]
for i in range(M+1):
    m1_feat.append('ypp'+str(i+1))
m1_feat_y=list(m1_feat)
m1_feat_y.append('y')

m2_feat = ['technical_11','technical_13','technical_20','technical_30','20-30']
m2_feat_y = list(m2_feat).append('y')

total_feat = list(m1_feat_y)
for feats in m2_feat:
    total_feat.append(feats)
total_feat.append('timestamp')
total_feat.append('id')

def make_features(df, M=4, cut = False):
    #modifies the dataframe in place?
    a,b = 3.9398846626281738, 9.2311916351318359
    time = -1
    if cut:
        time = df.iloc[df.shape[0]-1]['timestamp']
    new_df = df[df['timestamp']>(time-6)].copy()
    
    EMA = new_df['technical_20']-new_df['technical_30']
    EMA_prev = new_df.groupby('id').shift(1)['technical_20']-new_df.groupby('id').shift(1)['technical_30']
    t13_delta = new_df['technical_13']-new_df.groupby('id').shift(1)['technical_13']
    new_df['ypp1'] = a*t13_delta+b*EMA-b*EMA_prev
    new_df['20-30'] = new_df['technical_20']-new_df['technical_30']
    for i in range(M):
        new_df['ypp'+str(i+2)] = new_df.groupby('id').shift(i+2)['ypp1']
    #print(new_df.head())    
    return new_df

#Initialize enviroment and get make training data
import kagglegym
env = kagglegym.make()
observation = env.reset()
df_train = observation.train
#medians = df_train.median(axis=0)

#We simply drop all the NA features
df_clean = make_features(df_train, M, False)[total_feat]
#df_clean_m = make_features(df_train_m, M, False)
#df_clean = df_clean[(df_clean['y'] > low_y) & (df_clean['y'] < high_y)]

df_clean.dropna(inplace=True)
print('made features')


model1 = linear_model.LinearRegression()
model1.fit(df_clean[m1_feat],df_clean['y'])
r = np.sqrt(model1.score(df_clean[m1_feat].fillna(0),df_clean['y']))
print("Done training Model 1: r={}".format(r))

model2 = linear_model.Ridge(alpha)
model2.fit(df_clean[m2_feat],df_clean['y'])
r = np.sqrt(model2.score(df_clean[m2_feat].fillna(0),df_clean['y']))
print("Done training Model 2: r={}".format(r))


hist_feat = ['id','timestamp','technical_13','technical_20','technical_30']
#feature_list_y = ['id','timestamp','technical_13','technical_20','technical_30','y']
hist=make_features(df_train)[total_feat]
while True:
    target = observation.target
    feat = (observation.features)
    hist = hist.append(feat)
    hist = make_features(hist, M, True)[total_feat]
    s = target.shape[0]
    N = hist.shape[0]
    yp1 = np.clip(model1.predict(hist[N-s:][m1_feat].fillna(0)),low_y,high_y)
    yp2 = np.clip(model2.predict(hist[N-s:][m2_feat].fillna(0)),low_y,high_y)
    target['y']= A*yp1 + (1-A)*yp1
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(reward)

    observation, reward, done, info = env.step(target)
    if done:        
        print(info)
        break
