from keras import models, layers
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_probability as tfp

#from keras import backend as K




df = pd.read_csv("powerconsumption.csv", delimiter = ",")

train_features = df.iloc[:, :6]
values = df.iloc[:, 3]


train_list = []

for index, row in train_features.iterrows():
    item_index = 0

    for item in row:
        item_index += 1
        if item_index == 1:
            item = datetime.strptime(item, "%m/%d/%Y %H:%M")# użyć funkcji datetime.strptime() do zapisania całej daty
            item = item.timestamp()


            train_list.append(item)

        else:

            train_list.append(int(item))


train_set = np.array(train_list, dtype=np.float32)

values_list = []
for item in values:

    values_list.append(int(item))

train_labels = np.array(values_list, dtype=np.float32)

min_length = min(len(train_set), len(train_labels))
train_set = train_set[:min_length]
train_labels = train_labels[:min_length]

tr_dataset = tf.data.Dataset.from_tensor_slices((train_set, train_labels))

batch_size = 6

train_dataset = (
    tr_dataset.shuffle(buffer_size=100)
           .batch(batch_size)
           .prefetch(buffer_size=tf.data.AUTOTUNE)
)

dfp = pd.read_csv("powercons_testset.csv", delimiter = ",")

test_features = df.iloc[:, :6]
tvalues = df.iloc[:, 3]


test_list = []

for index, row in test_features.iterrows():
    item_index = 0

    for item in row:
        item_index += 1
        if item_index == 1:
            item = datetime.strptime(item, "%m/%d/%Y %H:%M")# użyć funkcji datetime.strptime() do zapisania całej daty
            item = item.timestamp()


            test_list.append(item)

        else:

            test_list.append(int(item))


test_set = np.array(test_list, dtype=np.float32)

tvalues_list = []
for item in tvalues:

    tvalues_list.append(int(item))

test_labels = np.array(tvalues_list, dtype=np.float32)

min_length = min(len(test_set), len(test_labels))
test_set = test_set[:min_length]
test_labels = test_labels[:min_length]

tr2_dataset = tf.data.Dataset.from_tensor_slices((test_set, test_labels))

batch_size = 6

test_dataset = (
    tr2_dataset.shuffle(buffer_size=100)
           .batch(batch_size)
           .prefetch(buffer_size=tf.data.AUTOTUNE)
)

noise = 1.0

def neg_log_likelihood(y_obs, y_pred, sigma=noise): # this function was modified with reduce_sum function from original version by krasserm
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return tf.reduce_sum(-dist.log_prob(y_obs))


predicting_model = models.Sequential([
layers.Input(shape=(1,)),
layers.Dense(26073, activation='relu'),
#layers.Dropout(rate=0.1),

layers.Dropout(rate=0.1),
layers.Dense(9, activation='relu'),
layers.Dense(1, activation='sigmoid')
])



print(type(predicting_model))
predicting_model.compile(optimizer="adam", metrics=['mse'], loss=neg_log_likelihood)
predicting_model.fit(train_dataset, epochs=25 )



predicting_model.summary()
#
predicting_model.evaluate(test_dataset)
predicting_model.evaluate(test_dataset)
predicting_model.evaluate(test_dataset)
#

