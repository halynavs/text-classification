import numpy as np
import pandas as pd
#preprocessing
import text_normalizer as tn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#evaluating
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def get_metrics(true_labels, predicted_labels):
    return np.round(metrics.accuracy_score(true_labels,predicted_labels),4), np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),4), np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),4),np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),4)




data_df = tn.create_df_from_input_labeled('labeled2')

# normalize data
norm_corpus, emoji, timecode = tn.normalize_corpus(corpus=data_df['Comment'], extract_timecodes=True,
                                                   special_char_removal=True, use_emoji=True,
                                                   repeated_characters_remover=True, text_lower_case=True,
                                                   stop_words_remover = True, text_lemmatization=True)
data_df['Clean_Comment'] = norm_corpus
data_df['Emoji'] = emoji
data_df['TimeCodes'] = timecode

print("Cleanned comments:\n",data_df['Clean_Comment'])
print("Data shape",data_df.shape)

# find empty documents in dataset and remove them
total_nulls = data_df[data_df.Clean_Comment.str.strip() == ''].shape[0]
print("Empty documents:", total_nulls)
print("Data shape before removing empty documents:", data_df.shape)
# use a simple pandas filter operation and remove all the records with no textual content
data_df = data_df[~(data_df.Clean_Comment.str.strip() == '')]
print("Data shape", data_df.shape)


# split data on train and test
train_corpus, test_corpus, train_label_nums, test_label_nums =\
                                 train_test_split(np.array(data_df['Clean_Comment']), np.array(data_df['Label']), test_size=0.33, random_state=42)
print("Train data shape", train_corpus.shape, '\n',"Test data shape",  test_corpus.shape)


# build BOW features on train articles
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=0.8)
tv_train_features = tv.fit_transform(train_corpus)
# transform test articles into features
tv_test_features = tv.transform(test_corpus)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

#use LogisticRegression, LinearSVC, SGDClassifier(SVC), RandomForestClassifier, GradientBoostingClassifier for classification


lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
lr.fit(tv_train_features, train_label_nums)
lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, train_label_nums, cv=5)
lr_tfidf_cv_mean_score = np.mean(lr_tfidf_cv_scores)
lr_tfidf_test_score = lr.score(tv_test_features, test_label_nums)


svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm.fit(tv_train_features, train_label_nums)
svm_tfidf_cv_scores = cross_val_score(svm, tv_train_features, train_label_nums, cv=5)
svm_tfidf_cv_mean_score = np.mean(svm_tfidf_cv_scores)
svm_tfidf_test_score = svm.score(tv_test_features, test_label_nums)


svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42)
svm_sgd.fit(tv_train_features, train_label_nums)
svmsgd_tfidf_cv_scores = cross_val_score(svm_sgd, tv_train_features, train_label_nums, cv=5)
svmsgd_tfidf_cv_mean_score = np.mean(svmsgd_tfidf_cv_scores)
svmsgd_tfidf_test_score = svm_sgd.score(tv_test_features, test_label_nums)



rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(tv_train_features, train_label_nums)
rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, train_label_nums, cv=5)
rfc_tfidf_cv_mean_score = np.mean(rfc_tfidf_cv_scores)
rfc_tfidf_test_score = rfc.score(tv_test_features, test_label_nums)


gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbc.fit(tv_train_features, train_label_nums)
gbc_tfidf_cv_scores = cross_val_score(gbc, tv_train_features, train_label_nums, cv=5)
gbc_tfidf_cv_mean_score = np.mean(gbc_tfidf_cv_scores)
gbc_tfidf_test_score = gbc.score(tv_test_features, test_label_nums)


# evaluation
# calculation confusion matrix and save it in xlsx
cm_lr = confusion_matrix(test_label_nums,lr.predict(tv_test_features))
cm_svm = confusion_matrix(test_label_nums,svm.predict(tv_test_features))
cm_svm_sgd = confusion_matrix(test_label_nums,svm_sgd.predict(tv_test_features))
cm_rfc =confusion_matrix(test_label_nums,rfc.predict(tv_test_features))
cm_gbc = confusion_matrix(test_label_nums,gbc.predict(tv_test_features))

df1=pd.DataFrame([['Predicted:',cm_lr[0][0],cm_lr[1][0]],
                    ['',cm_lr[0][1],cm_lr[1][1]]],
             columns=['Logistic Regression','Actual:', ''], ).T
df2=pd.DataFrame([['Predicted:',cm_svm[0][0],cm_svm[1][0]],
                    ['',cm_svm[0][1],cm_svm[1][1]]],
             columns=['Linear SVM','Actual:', ''],).T
df3=pd.DataFrame([['Predicted:',cm_svm_sgd[0][0],cm_svm_sgd[1][0]],
                    ['',cm_svm_sgd[0][1],cm_svm_sgd[1][1]]],
             columns=['Linear SVM (SGD)','Actual:', ''],).T
df4=pd.DataFrame([['Predicted:',cm_rfc[0][0],cm_rfc[1][0]],
                    ['',cm_rfc[0][1],cm_rfc[1][1]]],
             columns=['Random Forest','Actual:', ''],).T
df5=pd.DataFrame([['Predicted:',cm_gbc[0][0],cm_gbc[1][0]],
                    ['',cm_gbc[0][1],cm_gbc[1][1]]],
             columns=['Gradient Boosted Machines','Actual:', ''],).T

writer = pd.ExcelWriter('Confusion_matrix.xlsx', engine='xlsxwriter')
# write each dataframe to a different place on worksheet.
df1.to_excel(writer, sheet_name='confusion_matrix')
df2.to_excel(writer, sheet_name='confusion_matrix', startcol=4)
df3.to_excel(writer, sheet_name='confusion_matrix', startcol=8)
df4.to_excel(writer, sheet_name='confusion_matrix', startrow=6)
df5.to_excel(writer, sheet_name='confusion_matrix', startcol=4, startrow=6)
writer.save()

# calculation of Accuracy, Precision, Recall, F1 Score and save it in xlsx
lr_rate = get_metrics(true_labels=test_label_nums, predicted_labels=lr.predict(tv_test_features))
svm_rate = get_metrics(true_labels=test_label_nums, predicted_labels=svm.predict(tv_test_features))
svm_sgd_rate = get_metrics(true_labels=test_label_nums, predicted_labels=svm_sgd.predict(tv_test_features))
rfc_rate = get_metrics(true_labels=test_label_nums, predicted_labels=rfc.predict(tv_test_features))
gbc_rate = get_metrics(true_labels=test_label_nums, predicted_labels=gbc.predict(tv_test_features))

pd.DataFrame([['Logistic Regression', lr_rate[0], lr_rate[1],
               lr_rate[2], lr_rate[3]],
              ['Linear SVM', svm_rate[0], svm_rate[1],
               svm_rate[2], svm_rate[3]],
              ['Linear SVM (SGD)', svm_sgd_rate[0], svm_sgd_rate[1],
               svm_sgd_rate[2], svm_sgd_rate[3]],
              ['Random Forest', rfc_rate[0], rfc_rate[1],
               rfc_rate[2], rfc_rate[3]],
              ['Gradient Boosted Machines', gbc_rate[0], gbc_rate[1],
               gbc_rate[2], gbc_rate[3]]],
             columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
             ).T.to_excel('Metrics.xlsx')


# calculation of Mean cross validation score & test score of predictions
pd.DataFrame([['Logistic Regression',lr_tfidf_cv_mean_score, lr_tfidf_test_score],
              ['Linear SVM', svm_tfidf_cv_mean_score, svm_tfidf_test_score],
              ['Linear SVM (SGD)', svmsgd_tfidf_cv_mean_score, svmsgd_tfidf_test_score],
              ['Random Forest', rfc_tfidf_cv_mean_score, rfc_tfidf_test_score],
              ['Gradient Boosted Machines', gbc_tfidf_cv_mean_score, gbc_tfidf_test_score]],
             columns=['Model', 'CV Score (TF-IDF)', 'Test Score (TF-IDF)'],
             ).T.to_excel('Mean_cross_val_score&test_score.xlsx')


