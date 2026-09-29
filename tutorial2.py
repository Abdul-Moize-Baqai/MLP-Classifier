import warnings
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("scikit-learn version:", sklearn.__version__)
print("Training samples:", len(y_train), "| Test samples:", len(y_test))

# Each call creates a fresh model, rather than continuing an earlier run.
def train_model(layers, rate):
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        learning_rate_init=rate,
        max_iter=1000,
        random_state=42,
        tol=1e-4,
        n_iter_no_change=10,
        early_stopping=False,
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y_train)
    capped = any(issubclass(w.category, ConvergenceWarning) for w in recorded)
    return {
        "model": model,
        "train_accuracy": accuracy_score(y_train, model.predict(X_train_scaled)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test_scaled)),
        "epochs": model.n_iter_,
        "loss": model.loss_curve_[-1],
        "status": "Iteration limit" if capped else "Tolerance stop",
    }

def print_results(results):
    print(f'{"Setting":<17}{"Train %":>10}{"Test %":>10}{"Epochs":>9}{"Loss":>11}  Stop reason')
    for label, r in results.items():
        print(f'{label:<17}{100*r["train_accuracy"]:>10.2f}'
              f'{100*r["test_accuracy"]:>10.2f}{r["epochs"]:>9}'
              f'{r["loss"]:>11.4f}  {r["status"]}')

def plot_results(results, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, r in results.items():
        losses = r["model"].loss_curve_
        axes[0].plot(range(1, len(losses)+1), losses, label=label)
    axes[0].set(xlabel="Epoch (one pass through training data)",
                ylabel="Training loss", title="Loss during training")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    labels = list(results)
    positions = np.arange(len(labels))
    axes[1].bar(positions-0.18, [100*r["train_accuracy"] for r in results.values()],
                width=0.36, label="Training")
    axes[1].bar(positions+0.18, [100*r["test_accuracy"] for r in results.values()],
                width=0.36, label="Test")
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].set(ylabel="Accuracy (%)", ylim=(0, 105), title="Accuracy after training")
    axes[1].legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()
#Task 1 ( change hidden layer sizes)
architectures = [(5,), (10,), (50,), (10, 10), (10, 10, 10), (50, 50)]
architecture_results = {}
for layers in architectures:
    architecture_results[str(layers)] = train_model(layers, 0.001)
print_results(architecture_results)
plot_results(architecture_results, "Task 1: Architecture comparison | learning rate = 0.001")
#task 2 (change learning rate)
learning_rates = [0.0001, 0.001, 0.01, 0.1]
rate_results = {}
for rate in learning_rates:
    rate_results[str(rate)] = train_model((10, 10), rate)
print_results(rate_results)
plot_results(rate_results, "Task 2: Learning-rate comparison | hidden layers = (10, 10)")