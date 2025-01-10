import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report

def load_dataset():
    dataset = pd.read_csv('creditcard.csv', sep=',')
    return dataset

def show_head(dataset):
    print("Dataset head:")
    print(dataset.head())

def show_tail(dataset):
    print("Dataset tail:")
    print(dataset.tail())

def plot_transaction_distribution(dataset):
    x = dataset[dataset['Class'] == 1].shape[0]
    y = dataset[dataset['Class'] == 0].shape[0]
    plt.bar(['Fraud', 'Normal'], [x, y])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Transaction Distribution')
    plt.show(block=False)

def plot_histogram(dataset):
    fraudamount = dataset[dataset['Class'] == 1]['Amount']
    normalamount = dataset[dataset['Class'] == 0]['Amount']
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].hist(fraudamount, bins=50, color='red', alpha=0.7)
    axs[0].set_title('Fraud Transactions')
    axs[0].set_xlabel('Amount')
    axs[0].set_ylabel('Transaction')
    axs[1].hist(normalamount, bins=50, color='green', alpha=0.7)
    axs[1].set_title('Normal Transactions')
    axs[1].set_xlabel('Amount')
    axs[1].set_ylabel('Transaction')
    plt.tight_layout()
    plt.show(block=False)

def sample_dataset(dataset):
    new_dataset = dataset.sample(frac=0.2, random_state=1)
    return new_dataset

def train_isolation_forest(new_dataset):
    fraudcases = new_dataset[new_dataset['Class'] == 1].shape[0]
    normalcases = new_dataset[new_dataset['Class'] == 0].shape[0]
    fraction = fraudcases / float(normalcases)
    target = 'Class'
    col = new_dataset.columns.tolist()
    col = [c for c in col if c != 'Class']
    state = np.random.RandomState(42)
    X = new_dataset[col]
    Y = new_dataset[target]
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=fraction, random_state=state, verbose=0)
    model.fit(X)
    y_pred = model.predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return model, X, Y, y_pred

def evaluate_model(Y, y_pred):
    n_errors = (y_pred != Y).sum()
    accuracy = accuracy_score(Y, y_pred)
    print("Number of errors:", n_errors)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(Y, y_pred))

def menu():
    dataset = None
    new_dataset = None
    model = None
    X = None
    Y = None
    y_pred = None

    while True:
        print("\nMenu:")
        print("1. Load Dataset")
        print("2. Show Dataset Head")
        print("3. Show Dataset Tail")
        print("4. Plot Transaction Distribution")
        print("5. Plot Histogram")
        print("6. Sample Dataset")
        print("7. Train Isolation Forest Model")
        print("8. Evaluate Model")
        print("9. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            dataset = load_dataset()
            print("Dataset loaded successfully.")
        elif choice == '2':
            if dataset is not None:
                show_head(dataset)
            else:
                print("Load the dataset first.")
        elif choice == '3':
            if dataset is not None:
                show_tail(dataset)
            else:
                print("Load the dataset first.")
        elif choice == '4':
            if dataset is not None:
                plot_transaction_distribution(dataset)
            else:
                print("Load the dataset first.")
        elif choice == '5':
            if dataset is not None:
                plot_histogram(dataset)
            else:
                print("Load the dataset first.")
        elif choice == '6':
            if dataset is not None:
                new_dataset = sample_dataset(dataset)
                print("Original dataset shape:", dataset.shape)
                print("New dataset shape:", new_dataset.shape)
            else:
                print("Load the dataset first.")
        elif choice == '7':
            if new_dataset is not None:
                model, X, Y, y_pred = train_isolation_forest(new_dataset)
                print("Model trained successfully.")
            else:
                print("Sample the dataset first.")
        elif choice == '8':
            if model is not None and X is not None and Y is not None and y_pred is not None:
                evaluate_model(Y, y_pred)
            else:
                print("Train the model first.")
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    menu()
