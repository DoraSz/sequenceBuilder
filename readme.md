# Sequence Builder: transform your time-series-data into sequences to feed into an LSTM Neural Network

## Information

SequenceBuilder is a pip package that enables preprocessing time-series-data contained in pandas dataframes and creating sequential input for further usage in LSTM neural networks for forecasting applications.

## Setup

To install the package, run 

```
pip install sequence-builder

```

and add 

```python
from sequence_builder import Sequence_builder 
```

to your code.

You can also download the codebase from here and add 

```python
from Sequence_builder import Sequence_builder
```

to your code.

## Example

When working with time-series-data and neural networks, you may want to transform your data from linearly ordered single data points to sequences of ordered data points of specific length.

For example given your initial 4-dimensional data

```
A, B, C, D
1,10,100,1000
2,20,200,2000
3,30,300,3000
4,40,400,4000
5,50,500,5000
6,60,600,6000
7,70,700,7000
8,80,800,8000
9,90,900,9000
...
```

you can create training sequences of length 3 and of dimension 4 as

```
[1,10,100,1000],[2,20,200,2000],[3,30,300,3000]
[2,20,200,2000],[3,30,300,3000],[4,40,400,4000]
[3,30,300,3000],[4,40,400,4000],[5,50,500,5000]
[4,40,400,4000],[5,50,500,5000],[6,60,600,6000]
[5,50,500,5000],[6,60,600,6000],[7,70,700,7000]
[6,60,600,6000],[7,70,700,7000],[8,80,800,8000]
[7,70,700,7000],[8,80,800,8000],[9,90,900,9000]
...
```

To forecast future values (e.g. stock market prices, telemetry values), build your training data as follows (by shifting future values)  
```
					   X                                 Y
[1,10,100,1000],[2,20,200,2000],[3,30,300,3000]   [4,40,400,4000]
[2,20,200,2000],[3,30,300,3000],[4,40,400,4000]   [5,50,500,5000]
[3,30,300,3000],[4,40,400,4000],[5,50,500,5000]   [6,60,600,6000]
[4,40,400,4000],[5,50,500,5000],[6,60,600,6000]   [7,70,700,7000]
[5,50,500,5000],[6,60,600,6000],[7,70,700,7000]   [8,80,800,8000]
[6,60,600,6000],[7,70,700,7000],[8,80,800,8000]   [9,90,900,9000]
[7,70,700,7000],[8,80,800,8000],[9,90,900,9000]         ...


```

## Usage

To create sequences from your data, make sure your data is contained in a pandas dataframe with the appropriate header to address the columns.

Build you sequences as follows:

```python
"""
df: a pandas dataframe containing the training data
DIMENSIONS_TO_PREDICT: list of the features (column names) that you want to forecast
SEQ_LEN: length of the sequences we want to create
PREDICTION_LENGTH: how many consecutive steps you want to predict ahead
SHUFFLE_SEQUENCES: set to True if you want your training data to be randomly shuffled after the sequences are built (your training targets stay aligned with the shuffled data)
labels (optional): put in an extra dataframe with features (eg. class labels) you want to separate from your training data, but evaluate further after testing (your labels stay aligned with the shuffled data)
"""

seq_builder = Sequence_builder()
train_X, train_y, train_tags = seq_builder.fit_transform(df, DIMENSIONS_TO_PREDICT=['A','B','C','D'], 
SEQ_LEN=3, PREDICTION_LENGTH=1, SHUFFLE_SEQUENCES=False)
```

This example creates the sequences showed above. The sequence builder outputs multidimensional numpy arrays that you can use for example to feed into your Keras LSTM neural network as follows in this minimal example:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(optimizer='SGD', loss='mean_squared_error')
model.fit(train_X, train_y)
```