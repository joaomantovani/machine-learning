import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model


colnames = ["n_tokens_title", "n_tokens_content", "n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "average_token_length", "num_keywords", "data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world", "kw_min_min", "kw_max_min", "kw_avg_min", "kw_min_max", "kw_max_max", "kw_avg_max", "kw_min_avg", "kw_max_avg", "kw_avg_avg", "self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_sharess", "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday", "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", "weekday_is_sunday", "is_weekend", "LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04", "global_subjectivity", "global_sentiment_polarity", "global_rate_positive_words", "global_rate_negative_words", "rate_positive_words", "rate_negative_words", "avg_positive_polarity", "min_positive_polarity", "max_positive_polarity", "avg_negative_polarity", "min_negative_polarity", "max_negative_polarity", "title_subjectivity", "title_sentiment_polarity", "abs_title_subjectivity", "abs_title_sentiment_polarity"]

colnames = [ "global_rate_positive_words", "global_rate_negative_words", "rate_positive_words", "rate_negative_words", "avg_positive_polarity", "min_positive_polarity", "max_positive_polarity", "avg_negative_polarity"]




col_names = colnames
shares = ["shares"]

# Training dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train[col_names]
Y_train = dataset_train[shares]

# Testing dataset
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test[col_names]
# y_test = dataset_test[shares]

# Target Target Test
dataset_target = pd.read_csv('test_target.csv')
# X_target = dataset_target[col_names]
Y_target = dataset_target[shares].values
# print(dataset_target)

# Algorithm
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)

X_train = X_train[:10000]
Y_train = Y_train[:10000]

clf = linear_model.SGDRegressor()
results = clf.fit(X_train, Y_train.values.ravel())

y_pred = results.predict(X_test)

formated_y_target = list()
for y_tar in Y_target:
    formated_y_target.append(y_tar[0])

# formated_y_pred = list()
# for y_pred in y_pred:
#     formated_y_pred.append(y_pred[0])

formated_y_pred = y_pred

df = pd.DataFrame({'Actual': formated_y_target, 'Predicted': formated_y_pred})
df.to_csv('results.csv', sep='\t', encoding='utf-8')
print(df)

print('')
print('Mean Absolute Error:', metrics.mean_absolute_error(formated_y_target, formated_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(formated_y_target, formated_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(formated_y_target, formated_y_pred)))

#print('')
#print(dataset_train['shares'].describe())

plt.plot(formated_y_target, 'o', label='Esperado')
plt.plot(formated_y_pred, 'o', label='Predict')

# plt.plot(X_, pearson.intercept + pearson.slope*x, 'r', label='Tendencia')
plt.legend()
#plt.show()