# Tutorial 2: MLP Classifier

This project uses an MLP to classify flowers from the Iris dataset. The two tasks are to change the number of hidden layers and neurons, and then change the learning rate to see how these affect accuracy and training loss.

## Setup

The dataset contains 150 flowers from three classes: Setosa, Versicolor, and Virginica. Each flower has four measurements: sepal length, sepal width, petal length, and petal width.

The data was split into 70% training and 30% testing, giving 105 training samples and 45 test samples. StandardScaler was fitted on the training data and used to scale both sets.

The experiments used:

- ReLU activation in the hidden layers
- Adam optimizer
- `random_state=42` for the split and model
- `max_iter=1000`
- `tol=0.0001` and `n_iter_no_change=10`

The results below are from the accompanying tutorial notebook, run with scikit-learn 1.8.0. They should be checked against the output of `MLP Classifier.py` before comparing results.

## Task 1: Changing hidden layers and neurons

The learning rate was kept at `0.001` while the hidden-layer configuration was changed. For example, `(10,)` means one hidden layer with 10 neurons, while `(10, 10)` means two hidden layers with 10 neurons each.

| Hidden layers | Test accuracy | Epochs | Final training loss |
|---|---:|---:|---:|
| `(5,)` | 100% | 1000* | 0.1390 |
| `(10,)` | 100% | 1000* | 0.0947 |
| `(50,)` | 100% | 591 | 0.0735 |
| `(10, 10)` | 100% | 609 | 0.0718 |
| `(10, 10, 10)` | 100% | 857 | 0.0318 |
| `(50, 50)` | 100% | 353 | 0.0461 |

*These runs reached the maximum of 1000 epochs before satisfying the stopping criterion.*

### Observations

All configurations gave the same test accuracy, so adding more layers or neurons did not improve accuracy on this split. However, the loss curves and number of epochs were different.

The `(50, 50)` model reduced its loss quickly and stopped after 353 epochs, the fewest in this comparison. The three-layer model `(10, 10, 10)` reached the lowest final training loss, but took 857 epochs. This shows that a deeper network does not always need fewer epochs.

There was no single best configuration based on test accuracy. `(5,)` was the smallest model, while `(50, 50)` used the fewest epochs. Training time was not measured, so fewer epochs cannot be taken to mean the shortest runtime.

## Task 2: Changing the learning rate

For this task, the hidden layers were kept at `(10, 10)` and only `learning_rate_init` was changed.

| Learning rate | Training accuracy | Test accuracy | Epochs | Final training loss |
|---|---:|---:|---:|---:|
| `0.0001` | 75.24% | 73.33% | 1000* | 0.5507 |
| `0.001` | 97.14% | 100% | 609 | 0.0718 |
| `0.01` | 98.10% | 100% | 157 | 0.0494 |
| `0.1` | 100% | 95.56% | 159 | 0.0011 |

*Reached the maximum of 1000 epochs.*

### Observations

At `0.0001`, the loss decreased slowly. The model reached the epoch limit with a test accuracy of 73.33%, so this rate was too small to learn enough within the allowed epochs.

Increasing the rate to `0.001` gave 100% test accuracy in 609 epochs. At `0.01`, the model reached the same accuracy in only 157 epochs, and the loss curve dropped more quickly.

At `0.1`, the training loss became very small, although the curve showed some oscillation. Training accuracy reached 100%, but test accuracy dropped to 95.56%. Fitting the training data better did not give better predictions on the test data.

Of the rates tested, `0.01` gave the best balance between test accuracy and the number of epochs.

## Conclusion

Changing the learning rate made a clearer difference than increasing the network size in these experiments. More layers and neurons helped reduce training loss, but did not improve test accuracy. The original `(10, 10)` model with a learning rate of `0.01` achieved 100% test accuracy in 157 epochs.

These findings apply to this particular split and random seed. The test set contains only 45 flowers, so 100% accuracy here does not mean the model will classify every new flower correctly. A separate validation set or cross-validation would be needed for a stronger comparison of the settings.
