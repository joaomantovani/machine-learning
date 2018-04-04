import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

np.random.seed(19680801)

#First test
colnames = ["url", "timedelta", "n_tokens_title", "n_tokens_content", "n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "average_token_length", "num_keywords", "data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world", "kw_min_min", "kw_max_min", "kw_avg_min", "kw_min_max", "kw_max_max", "kw_avg_max", "kw_min_avg", "kw_max_avg", "kw_avg_avg", "self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_sharess", "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday", "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", "weekday_is_sunday", "is_weekend", "LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04", "global_subjectivity", "global_sentiment_polarity", "global_rate_positive_words", "global_rate_negative_words", "rate_positive_words", "rate_negative_words", "avg_positive_polarity", "min_positive_polarity", "max_positive_polarity", "avg_negative_polarity", "min_negative_polarity", "max_negative_polarity", "title_subjectivity", "title_sentiment_polarity", "abs_title_subjectivity", "abs_title_sentiment_polarity", "shares"]

data = pandas.read_csv('train.csv', names=colnames)

cont = 1
muito_forte = 0
forte = 0
moderada = 0
fraca = 0
desprezivel = 0

arr = {}

for col in colnames:
    if cont == 60:
        break

    #X
    #x = data.global_rate_positive_words.tolist()
    x = getattr(data, colnames[cont])
    del x[0]
    x = np.array(x)
    x = np.asfarray(x, float)

    #Y
    y = data.shares.tolist()
    del y[0]
    y = np.array(y)
    y = np.asfarray(y,float)

    print("")
    print("Variavel #", cont, ": ", colnames[cont])
    pearson = linregress(x, y)
    cr = pearson.rvalue;

    arr[colnames[cont]] = cr

    if (cr > 0.9): #or (cr > -0.9)
        print("Correlacao: Muito Forte")
        muito_forte += 1

    if (cr > 0.7 and cr < 0.9): #or (cr > -0.7 and cr < -0.9)
        print("Correlacao: Forte")
        forte += 1

    if (cr > 0.5 and cr < 0.7): #or (cr > -0.5 and cr < -0.7)
        print("Correlacao: Moderada")
        moderada += 1

    if (cr > 0.3 and cr < 0.5): #or (cr > -0.3 and cr < -0.5)
        print("Correlacao: fraca")
        fraca += 1

    if cr > -0.3 and cr < 0.3:
        print("Correlacao: Desprezivel")
        desprezivel += 1

    #0.9 para mais ou para menos indica uma correlação muito forte.
    #0.7 a 0.9 positivo ou negativo indica uma correlação forte.
    #0.5 a 0.7 positivo ou negativo indica uma correlação moderada.
    #0.3 a 0.5 positivo ou negativo indica uma correlação fraca.
    #0 a 0.3 positivo ou negativo indica uma correlação desprezível.
    print(pearson)

    # plt.plot(x, y, 'o', label='Observacao')
    # plt.plot(x, pearson.intercept + pearson.slope*x, 'r', label='Tendencia')
    # plt.legend()
    # plt.show()

    cont += 1

print('')
print('')
# print('Contagens:')
# print('Muito Forte: ', muito_forte)
# print('Forte: ', forte)
# print('Moderada: ', moderada)
# print('Fraca: ', fraca)
# print('Desprezível: ', desprezivel)

sorted = sorted(arr.items(), key=lambda x: x[1], reverse=True)
print(sorted)