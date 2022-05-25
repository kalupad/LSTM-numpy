# LSTM-numpy

Long Short-Term Memory implementation with a plain Numpy package to analyze different types of sequential data.

Idea was generated thanks to Andrej Karpathy blog: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## Example prediction - semantic dataset:

```
Input:
['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

Target:
['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'END']

Prediction:
['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'END']
```

## Example prediction - numeric dataset:

```
Input:
[7, 8, 9, 10, 11, 12, 13]

Target:
[8, 9, 10, 11, 12, 13, 14]

Prediction:
[8, 9, 10, 11, 12, 13, 14]
```
