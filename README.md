# text-classification
## The program classifies russian comments into two classes: positive and negative.
### LogisticRegression, LinearSVC, SGDClassifier(SVC), RandomForestClassifier, GradientBoostingClassifier are used for classification.



Used data looks(warning: may contain strong language content):
![alt text](https://github.com/halynavs/text-classification/blob/main/screenshots/data.png)


### Module text_normalizer.py contains:
* function ***create_df_from_input_labeled(name_of_input_file)*** that from csv file makes pandas DataFrame
* and funtions for cleaning and normalazing data, which are used in **normalize_corpus** function.

Created df from input
![alt text](https://github.com/halynavs/text-classification/blob/main/screenshots/DF.png)

Function **normalize_corpus** takes data(column of comments) and parameters(all True by defauld)

```python
def normalize_corpus(corpus, extract_timecodes=True,
                     special_char_removal=True, use_emoji=True,
                     repeated_characters_remover=True, 
                     text_lower_case=True,
                     stop_words_remover = True, 
                     stopwords=stopword_list, 
                     text_lemmatization=True):
```
* if extract_timecodes = True creates list with timecode in comment or Null, else nothing adds
* if special_char_removal = True from data deletes everything exсept letters written in cyrillic
* if use_emoji = True creates list with emoji in comment or Null, else nothing adds
* repeated_characters_remover=True - remove repeated character( "привеееет"-->"привет")
* text_lower_case=True - lowercase conversion 
* stop_words_remover = True, stopwords=stopword_list - help to remove stop words that in stopword_list
* text_lemmatization=True - lemmatizate text, StanzaLanguage are used

After that clean text are added in df(as well Emoji and TimeCodes which can be used for better classification in the future ) 
```python
df['Clean_Comment'] = norm_corpus
df['Emoji'] = emoji
df['TimeCodes'] = timecode
```

### For classificasion are used cleaned data from the column of the same name 


1. Data splitted by sklearn.model_selection - train_test_split, proportion 33/77 - test/train
2. Text transformed into features(TFIDF) by sklearn.feature_extraction.text - TfidfVectorizer
3. Classificators are used with mantioned parameters: <br />
lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42) <br />
svm = LinearSVC(penalty='l2', C=1, random_state=42) <br />
svm_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42) <br />
rfc = RandomForestClassifier(n_estimators=10, random_state=42) <br />
gbc = GradientBoostingClassifier(n_estimators=10, random_state=42) <br />

### Estimations
Confusion matrix <br />  <br /> 
<img src="https://github.com/halynavs/text-classification/blob/main/screenshots/ConfusionMatrix.png" width="700"/> <br />  <br />   <br /> 
Metrics <br /> <br />
<img src="https://github.com/halynavs/text-classification/blob/main/screenshots/Metrics.png" width="500"/> <br /> <br />  <br /> 
Mean_cross_val_score&test_score  <br /> <br /> 
<img src="https://github.com/halynavs/text-classification/blob/main/screenshots/CrossValMetrics.png" width="600"/> <br /> <br />  <br /> 
