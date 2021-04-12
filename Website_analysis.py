import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("KCM Survey Website Preprocessed 1.0.csv",encoding = "ISO-8859-1",header=0)
conditions = [ (df['WEBSITE_GRADE'] >=8 ), (df['WEBSITE_GRADE'] >= 5) & (df['WEBSITE_GRADE'] < 8), (df['WEBSITE_GRADE'] <5)]
values_int = [2, 1, 0]
values_desc=['happy','good','sad']
df['grade_emotion']=np.select(conditions, values_int)
df['grade_emotion_desc']=np.select(conditions, values_desc)
df['GRADE_DETERMINING_FACTOR_PRESET'].replace({'INFO_FINDABILITY':'FINDABILITY','OTHER':'OTHER','CONVENIENCE_REPORT_OR_APPLICATION':'CONV_REP_APP',
 'CONVENIENCE_DETAILS_OR_STATUS':'CONV_DET_STAT','INFO_CLARITY':'CLARITY','WEBSITE_DESIGN':'WEB_DESIGN','WEBSITE_LOADING_TIME':'WEB_LOADTIME'}, inplace=True)
cat=df.groupby(['GRADE_DETERMINING_FACTOR_PRESET']).size()
cat_emo=df.groupby(['GRADE_DETERMINING_FACTOR_PRESET','grade_emotion_desc']).size()
'''
############### emotions per each category ###############
ax = cat.plot(kind='bar', figsize=(10,6), color="indigo", fontsize=13)
plt.xticks(rotation=40)
plt.show()
cat_emo.unstack().plot.bar(rot=0)
plt.show()
############### Correlation by heatmap ###################
#print(df.columns)
df_heatmap=df.drop(['grade_emotion','WEBSITE_GRADE','EXPLANATION','SUCCEEDED','IMPROVEMENT_TIPS','WHY_NOT_SUCCEEDED','Unnamed: 0'], axis = 1)
#get correlations of each features in dataset
corrmat = df_heatmap.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_heatmap[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.xticks(rotation=60)
plt.show()
'''
############### findability Correlation by heatmap if emotion is sad ###################
"""
df_corr=df[df['grade_emotion']==0].drop(['grade_emotion','WEBSITE_GRADE','EXPLANATION','SUCCEEDED','IMPROVEMENT_TIPS','WHY_NOT_SUCCEEDED','Unnamed: 0'], axis = 1)
#get correlations of each features in dataset
corrmat = df_corr.corr()
top_corr_features = corrmat.index
df_cor=df_corr[top_corr_features].corr()
df_findability=df_cor['INFO_FINDABILITY']
df_conv_rep_app=df_cor['CONVENIENCE_REPORT_OR_APPLICATION']
df_other=df_cor['OTHER']
df_web_loadpage=df_cor['WEBSITE_LOADING_TIME']
print(df_other)

plt.subplot(411)
ax1=df_findability.plot(kind='bar', color="indigo", fontsize=6)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.title('findability')
plt.ylabel('Importance')
plt.subplot(412)
ax2=df_conv_rep_app.plot(kind='bar', color="green", fontsize=6)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.title('Cov_rep_app')
plt.ylabel('Importance')
plt.subplot(413)
ax3=df_web_loadpage.plot(kind='bar', color="blue", fontsize=6)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.title('web_loadpage')
plt.ylabel('Importance')
plt.subplot(414)
ax4=df_other.plot(kind='bar', color="red", fontsize=6)
plt.title('other')
plt.ylabel('Importance')
plt.xticks(rotation=40)
plt.suptitle('Importance of features on categories with highest sadness feedback')
plt.show()
"""
################### Extracting important features ###################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing
import math
from sklearn.metrics import roc_auc_score

df_fi=df.drop(['ORIGIN','grade_emotion','WEBPAGE','WEBSITE_GRADE','EXPLANATION','SUCCEEDED','IMPROVEMENT_TIPS','WHY_NOT_SUCCEEDED','Unnamed: 0',
 'grade_emotion_desc','FILLED_ON'], axis = 1)
##################### preprocessing data ###############
df_fi['DEVICE_TYPE'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['DEVICE_TYPE'].unique().tolist())
X_DEVICE_TYPE=le.transform(df_fi['DEVICE_TYPE'])
df_fi['DEVICE'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['DEVICE'].unique().tolist())
X_DEVICE=le.transform(df_fi['DEVICE'])
df_fi['OS'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['OS'].unique().tolist())
X_OS=le.transform(df_fi['OS'])
df_fi['BROWSER'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['BROWSER'].unique().tolist())
X_BROWSER=le.transform(df_fi['BROWSER'])
df_fi['GRADE_FACTORS'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['GRADE_FACTORS'].unique().tolist())
X_GRADE_FACTORS=le.transform(df_fi['GRADE_FACTORS'])
df_fi['GRADE_DETERMINING_FACTOR'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['GRADE_DETERMINING_FACTOR'].unique().tolist())
X_GRADE_DETER_FACTOR=le.transform(df_fi['GRADE_DETERMINING_FACTOR'])
df_fi['VISIT_REASON'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['VISIT_REASON'].unique().tolist())
X_VISIT_REASON=le.transform(df_fi['VISIT_REASON'])
df_fi['FOUND_INFO'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['FOUND_INFO'].unique().tolist())
X_FOUND_INFO=le.transform(df_fi['FOUND_INFO'])
df_fi['SUBMIT_ONE_GO'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['SUBMIT_ONE_GO'].unique().tolist())
X_SUBMIT_ONE_GO=le.transform(df_fi['SUBMIT_ONE_GO'])
df_fi['PREVIOUS_REQUEST_INFO'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['PREVIOUS_REQUEST_INFO'].unique().tolist())
X_PREV_REQ_INFO=le.transform(df_fi['PREVIOUS_REQUEST_INFO'])
df_fi['INFO_FINDABILITY'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['INFO_FINDABILITY'].unique().tolist())
X_INFO_FINDABILITY=le.transform(df_fi['INFO_FINDABILITY'])
df_fi['INFO_CLARITY'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['INFO_CLARITY'].unique().tolist())
X_INFO_CLARITY=le.transform(df_fi['INFO_CLARITY'])
df_fi['CONVENIENCE_REPORT_OR_APPLICATION'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['CONVENIENCE_REPORT_OR_APPLICATION'].unique().tolist())
X_CONV_REPORT_OR_APP=le.transform(df_fi['CONVENIENCE_REPORT_OR_APPLICATION'])
df_fi['CONVENIENCE_DETAILS_OR_STATUS'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['CONVENIENCE_DETAILS_OR_STATUS'].unique().tolist())
X_CONV_DETAILS_OR_STAT=le.transform(df_fi['CONVENIENCE_DETAILS_OR_STATUS'])
df_fi['WEBSITE_DESIGN'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['WEBSITE_DESIGN'].unique().tolist())
X_WEBSITE_DESIGN=le.transform(df_fi['WEBSITE_DESIGN'])
df_fi['WEBSITE_LOADING_TIME'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['WEBSITE_LOADING_TIME'].unique().tolist())
X_WEBSITE_LOADING_TIME=le.transform(df_fi['WEBSITE_LOADING_TIME'])
df_fi['OTHER'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['OTHER'].unique().tolist())
X_OTHER=le.transform(df_fi['OTHER'])
df_fi['GRADE_DETERMINING_FACTOR_PRESET'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['GRADE_DETERMINING_FACTOR_PRESET'].unique().tolist())
X_GRADE_DETER_FACTOR_PRESET=le.transform(df_fi['GRADE_DETERMINING_FACTOR_PRESET'])
df_fi['GRADE_DETERMINING_FACTOR_OTHER'].fillna('Null', inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df_fi['GRADE_DETERMINING_FACTOR_OTHER'].unique().tolist())
X_GRADE_DETER_FACTOR_OTHER=le.transform(df_fi['GRADE_DETERMINING_FACTOR_OTHER'])
X_new=pd.DataFrame(columns=['DEVICE_TYPE', 'DEVICE', 'OS', 'BROWSER', 'VISIT_REASON', 
'FOUND_INFO', 'PREVIOUS_REQUEST_INFO', 'EFFORT_COST','INFO_FINDABILITY', 'INFO_CLARITY', 'CONV_REP_OR_APP',
'CONV_DETAILS_OR_STATUS', 'WEBSITE_DESIGN','WEBSITE_LOADING_TIME','X_GRADE_DETER_FACTOR_OTHER'])
X_new['DEVICE_TYPE'] = pd.Series(X_DEVICE_TYPE)
X_new['DEVICE'] = pd.Series(X_DEVICE)
X_new['OS'] = pd.Series(X_OS)
X_new['BROWSER'] = pd.Series(X_BROWSER)
#X_new['GRADE_FACTORS'] = pd.Series(X_GRADE_FACTORS)
#X_new['GRADE_DETERMINING_FACTOR'] = pd.Series(X_GRADE_DETER_FACTOR)
X_new['VISIT_REASON'] = pd.Series(X_VISIT_REASON)
X_new['FOUND_INFO'] = pd.Series(X_FOUND_INFO)
#X_new['SUBMIT_ONE_GO'] = pd.Series(X_SUBMIT_ONE_GO)
X_new['PREVIOUS_REQUEST_INFO'] = pd.Series(X_PREV_REQ_INFO)
X_new['EFFORT_COST'] = df_fi['CONVENIENCE_REPORT_OR_APPLICATION'] 
X_new['INFO_FINDABILITY'] = pd.Series(X_INFO_FINDABILITY)
X_new['INFO_CLARITY'] = pd.Series(X_INFO_CLARITY)
X_new['CONV_REP_OR_APP'] = pd.Series(X_CONV_REPORT_OR_APP)
X_new['CONV_DETAILS_OR_STATUS'] = pd.Series(X_CONV_DETAILS_OR_STAT)
X_new['WEBSITE_DESIGN'] = pd.Series(X_WEBSITE_DESIGN)
X_new['WEBSITE_LOADING_TIME'] = pd.Series(X_WEBSITE_LOADING_TIME)
#X_new['OTHER'] = pd.Series(X_OTHER)
#X_new['X_GRADE_DETER_FACTOR_PRESET'] = pd.Series(X_GRADE_DETER_FACTOR_PRESET)
X_new['X_GRADE_DETER_FACTOR_OTHER'] = pd.Series(X_GRADE_DETER_FACTOR_OTHER)
###########################################################
#Tfidf_vect = TfidfVectorizer(max_features=5000)
#Tfidf_vect.fit(df_fi)
#Train_X_Tfidf = Tfidf_vect.transform(X)
#print(Train_X_Tfidf.shape)
y = df[['grade_emotion']]
X = X_new
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.8, random_state = 42)
#set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
#print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(m.score(X_train, y_train),m.oob_score_,m.score(X_valid, y_valid)))
##R^2 Training Score: 0.68
##OOB Score: 0.20
##R^2 Validation Score: 0.25
fi = pd.DataFrame({'feature': X_train.columns, 'importance': m.feature_importances_}).sort_values(by='importance', ascending=False)
fi = fi.reset_index()
#print(fi)
#ax = fi[['feature', 'importance']].plot(kind='bar', figsize=(10,6), color="indigo", fontsize=13)
#plt.xticks(rotation=40)
#plt.show()
'''
   index                     feature  importance
0       5                  FOUND_INFO    0.254089
1       9                INFO_CLARITY    0.181886
2       3                     BROWSER    0.112322
3       4                VISIT_REASON    0.060209
4       8            INFO_FINDABILITY    0.057450
5      13        WEBSITE_LOADING_TIME    0.050931
6       2                          OS    0.044328
7      12              WEBSITE_DESIGN    0.042978
8      11      CONV_DETAILS_OR_STATUS    0.042672
9       1                      DEVICE    0.033394
10      0                 DEVICE_TYPE    0.026967
11      6       PREVIOUS_REQUEST_INFO    0.024418
12     10             CONV_REP_OR_APP    0.024312
13      7                 EFFORT_COST    0.023646
14     14  X_GRADE_DETER_FACTOR_OTHER    0.020398
'''
df_group=df[['FOUND_INFO','OS','INFO_CLARITY','BROWSER','VISIT_REASON','INFO_FINDABILITY','WEBSITE_LOADING_TIME','WEBSITE_DESIGN','WEBPAGE','ORIGIN',
'grade_emotion','grade_emotion_desc']]
df_group['WEBPAGE'].fillna(df_group['ORIGIN'], inplace=True)
df_group['WEBPAGE'].fillna('Null', inplace=True)
df_group['FOUND_INFO'].fillna('Null', inplace=True)
df_group['INFO_CLARITY'].fillna('Null', inplace=True)
df_group['BROWSER'].fillna('Null', inplace=True)
df_group['VISIT_REASON'].fillna('Null', inplace=True)
df_group['INFO_FINDABILITY'].fillna('Null', inplace=True)
df_group['WEBSITE_LOADING_TIME'].fillna('Null', inplace=True)
df_group['WEBSITE_DESIGN'].fillna('Null', inplace=True)
#print(df_group.columns)
df_group['WEBPAGE1']=df_group['WEBPAGE'].replace(to_replace ='https://www.rvo.nl/',  value ='', regex = True)
df_group['web_subCat']=df_group['WEBPAGE1'].replace(to_replace ='zoeken.+', value='zoeken',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='/.+', value='',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='subsidie-en-financieringswijzer.+', value='subsidie-en-financieringswijzer',regex = True)##135 categories
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='subsidies-regelingen.+', value='subsidies-regelingen',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='coronavirus.+', value='coronavirus',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='vergunningen-online.+', value='vergunningen-online',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='documenten-publicaties.+', value='documenten-publicaties',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='octrooiportal.+', value='octrooiportal',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='contactformulier-rijksdienst-voor-ondernemend-nederland-rvonl-wssl.+',
 value='contactformulier-rijksdienst-voor-ondernemend-nederland-rvonl-wssl',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='contactformulier-wssl.+', value='contactformulier-wssl',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='financiering-voor-ondernemers.+', value='financiering-voor-ondernemers',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='utm_campaign=.+', value='utm_campaign',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='financiering-voor-internationaal-ondernemen.+', value='financiering-voor-internationaal-ondernemen',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='onderwerpen.+', value='onderwerpen',regex = True)
df_group['web_subCat']=df_group['web_subCat'].replace(to_replace ='', value='rvo.nl',regex = True)

#print(df_group['web_subCat'].nunique())### distinct 63 sub Folder
################ analysis of subfolders by most important features ###########
### INFO_FINDABILITY---
NoFind_Sad=df_group[(df_group['FOUND_INFO']=='No') & (df_group['grade_emotion']==0)].groupby(['web_subCat']).size().nlargest(10) 
#print("Sub folders with Sadness about no finding info : " , NoFind_Sad )
### INFO_CLARITY
NoClarity_Sad=df_group[(df_group['INFO_CLARITY']==True) & (df_group['grade_emotion']==0)].groupby(['web_subCat']).size().nlargest(10) 
#print("Sub folders with Sadness about no clarity info : " ,NoClarity_Sad)
WebLoading_Sad=df_group[(df_group['WEBSITE_LOADING_TIME']==True) & (df_group['grade_emotion']==0)].groupby(['web_subCat']).size().nlargest(10) 
#print("Sub folders with Sadness about loading webpage : " ,WebLoading_Sad)
WebDesign_Sad=df_group[(df_group['WEBSITE_DESIGN']==True) & (df_group['grade_emotion']==0)].groupby(['web_subCat']).size().nlargest(10) 
#print("Sub folders with Sadness about Web design : " ,WebDesign_Sad)
Browser_Sad=df_group[df_group['grade_emotion']==0].groupby(['web_subCat','BROWSER']).size().nlargest(10) 
#print("Sub folders with Sadness and browser : " ,Browser_Sad)
visitReason_Sad=df_group[df_group['grade_emotion']==0].groupby(['web_subCat','VISIT_REASON']).size().nlargest(10) 
#print("Sub folders with Sadness and visit reason : " ,visitReason_Sad)
OS_Sad=df_group[df_group['grade_emotion']==0].groupby(['web_subCat','OS']).size().nlargest(10) 
#print("Sub folders with Sadness and OS : " ,OS_Sad)

########################### happiness ##########################
### INFO_FINDABILITY---
NoFind_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','FOUND_INFO']).size().nlargest(10) 
#print("Sub folders with happiness about no finding info : " , NoFind_Sad )
### INFO_CLARITY
NoClarity_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','INFO_CLARITY']).size().nlargest(10) 
#print("Sub folders with happiness about no clarity info : " ,NoClarity_Sad)
WebLoading_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','WEBSITE_LOADING_TIME']).size().nlargest(10) 
#print("Sub folders with happiness about loading webpage : " ,WebLoading_Sad)
WebDesign_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','WEBSITE_DESIGN']).size().nlargest(10) 
#print("Sub folders with happiness about Web design : " ,WebDesign_Sad)
Browser_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','BROWSER']).size().nlargest(30) 
print("Sub folders with happiness and browser : " ,Browser_Sad)
visitReason_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','VISIT_REASON']).size().nlargest(10) 
#print("Sub folders with happiness and visit reason : " ,visitReason_Sad)
OS_Sad=df_group[df_group['grade_emotion']==2].groupby(['web_subCat','OS']).size().nlargest(10) 
#print("Sub folders with happiness and OS : " ,OS_Sad)
################ Encoding features to cluster##############
'''
df_cl=df_group[['web_subCat','FOUND_INFO','INFO_CLARITY','BROWSER','VISIT_REASON','INFO_FINDABILITY','WEBSITE_LOADING_TIME','WEBSITE_DESIGN']]
le = preprocessing.LabelEncoder()
le.fit(df_cl['web_subCat'].unique().tolist())
Y_web_subCat=le.transform(df_cl['web_subCat'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['FOUND_INFO'].unique().tolist())
Y_FOUND_INFO=le.transform(df_cl['FOUND_INFO'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['INFO_CLARITY'].unique().tolist())
Y_INFO_CLARITY=le.transform(df_cl['INFO_CLARITY'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['BROWSER'].unique().tolist())
Y_BROWSER=le.transform(df_cl['BROWSER'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['VISIT_REASON'].unique().tolist())
Y_VISIT_REASON=le.transform(df_cl['VISIT_REASON'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['INFO_FINDABILITY'].unique().tolist())
Y_INFO_FINDABILITY=le.transform(df_cl['INFO_FINDABILITY'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['WEBSITE_LOADING_TIME'].unique().tolist())
Y_WEBSITE_LOADING_TIME=le.transform(df_cl['WEBSITE_LOADING_TIME'])
le = preprocessing.LabelEncoder()
le.fit(df_cl['WEBSITE_DESIGN'].unique().tolist())
Y_WEBSITE_DESIGN=le.transform(df_cl['WEBSITE_DESIGN'])
X_cl=pd.DataFrame(columns=['web_subCat','FOUND_INFO','INFO_CLARITY','BROWSER',
'VISIT_REASON','INFO_FINDABILITY','WEBSITE_LOADING_TIME','WEBSITE_DESIGN'])
X_cl['web_subCat'] = pd.Series(Y_web_subCat)
X_cl['FOUND_INFO'] = pd.Series(Y_FOUND_INFO)
X_cl['INFO_CLARITY'] = pd.Series(Y_INFO_CLARITY)
X_cl['BROWSER'] = pd.Series(Y_BROWSER)
X_cl['VISIT_REASON'] = pd.Series(Y_VISIT_REASON)
X_cl['INFO_FINDABILITY'] = pd.Series(Y_INFO_FINDABILITY)
X_cl['WEBSITE_LOADING_TIME'] = pd.Series(Y_WEBSITE_LOADING_TIME)
X_cl['WEBSITE_DESIGN'] = pd.Series(Y_WEBSITE_DESIGN)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_cl)
print(kmeans.cluster_centers_)
'''
