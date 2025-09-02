import numpy as np
import matplotlib.pyplot as plt


def predict_value(alpha, actual_values, addition_fields):
    predicted_values = [actual_values[0]]
    T = len(actual_values)
    for t in range(1, T):
        LHS = 0.0
        for j, value in enumerate(actual_values[0:t]):
            LHS += alpha * ((1 - alpha) ** j) * actual_values[t - (j+1)]
        RHS = ((1 - alpha) ** t) * predicted_values[0]
        predicted_values.append(LHS + RHS)
    for i in range(addition_fields):
        predicted_values.append(predicted_values[-1])
    return np.array(predicted_values)

def ses(alpha, actual_values):
    predicted_values = [actual_values[0]]
    T = len(actual_values)
    for t in range(0, T-1):
        predicted_values.append(alpha * actual_values[t] + (1 - alpha) * predicted_values[t])
    return np.array(predicted_values)


if __name__ == "__main__":
    filename = "C:\SharedVM\TimeSeries\LAB2\AlgeriaExport.txt"
    data = []
    with open(filename, newline='\n') as f:
        for line in f:
            line = line.strip()
            data.append(float(line))
    data = np.array(data)
    division = int(2 * len(data) / 3)
    train_set = data[:division]
    test_set = data[division:]
    possible_alphas = np.linspace(0, 1, 100)
    min_SSE = np.inf
    min_alpha = np.inf
    for alpha in possible_alphas:
        SSE = 0.0
        predicted_value = predict_value(alpha, train_set, 0)
        SSE = train_set - predicted_value
        SSE = np.square(SSE)
        SSE = np.sum(SSE)
        if SSE < min_SSE:
            min_alpha = alpha
            min_SSE = SSE
    print(min_alpha)
    predicted_value = predict_value(min_alpha, train_set, len(data) - division)
    plt.plot(data, label="Actual data")
    plt.plot(predicted_value, label="Predicted")
    plt.legend()
    plt.ylabel("% of GDP")
    plt.xlabel("Year")
    plt.grid()
    plt.title("Algeria exports")
    plt.axvline(x=division-1, c='r', linestyle='--')
    plt.show()
    plt.plot(data, label="Actual data")
    plt.legend()
    plt.ylabel("% of GDP")
    plt.xlabel("Year")
    plt.grid()
    plt.title("Algeria exports")
    plt.show()

    possible_alphas = np.linspace(0, 1, 100)
    min_SSE = np.inf
    min_alpha = np.inf
    for alpha in possible_alphas:
        SSE = 0.0
        predicted_value = ses(alpha, train_set)
        SSE = train_set - predicted_value
        SSE = np.square(SSE)
        SSE = np.sum(SSE)
        if SSE < min_SSE:
            min_alpha = alpha
            min_SSE = SSE
    print(min_alpha)
    train_set = data[:40]
    test_set = data[division:]
    pred = ses(0.5, train_set)
    print(np.sum(np.square(train_set - pred)))