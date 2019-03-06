# Sequence Builder: transform your time-series-data into sequences to feed into an LSTM Neural Network

## Information

SequenceBuilder is a pip package/python class that enables preprocessing time-series-data contained in pandas dataframes and creating sequential input for further usage in LSTM neural networks for forecasting applications or classification tasks.

## Setup

To install the package, run 

```
pip install sequence_builder
```

and add 

```python
import sequence_builder
```

to your code.

You can also download the codebase from here and add 

```python
from Sequence_builder import Sequence_builder
```

to your code.

## Example

When working with time-series-data, you may want to transform your data from linearly ordered single data points to sequences of ordered data points of specific length.

For example given your intial 4-dimensional data

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

you can create training sequences of length 3 and dimension of 4 as

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

To carry out classification tasks, you can fill Y with class labels instead of forecasted values.

```

## Usage

To create sequences from your data, make sure your data is contained in pandas dataframes.

Build you sequences as follows:

```python
"""
df: a pandas dataframe containing the training data
DIMENSIONS_TO_PREDICT: list of the column names that we want to forecast (training targets)
SEQ_LEN: length of sequences to create
SHUFFLE_SEQUENCES: set to True if you want your data to be randomly shuffled after the sequences are built
labels: put in an extra dataframe with features you want to keep out of your data 
but evaluate further after testing
"""
seq_builder = Sequence_builder()
train_X, train_y, train_tags = seq_builder.fit_transform(df, DIMENSIONS_TO_PREDICT=['A','B','C','D'], 
SEQ_LEN=3, PREDICTION_LENGTH=1, SHUFFLE_SEQUENCES=True, labels=tags)
```

This example creates the example above.
