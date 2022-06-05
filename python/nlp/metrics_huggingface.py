# f1
def compute_metrics(eval_pred):

  predictions, references = eval_pred
  predictions = np.argmax(predictions, axis=1) # multi-classification

  # 'micro', 'macro', etc. are for multi-label classification. If you are running a binary classification, leave it as default or specify "binary" for average
  return metric.compute(predictions=predictions, references=references, average="micro")  


# rmse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    print("Logits:", logits[0:5])
    print("Labels:", labels[0:5])
    # print("Labels:", labels)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    accuracy = sum([1 for e in single_squared_errors if e < 0.5]) / len(single_squared_errors)

    return {"mse": mse, "rmse": rmse, "mae": mae, "smape": smape, "r2": r2, "accuracy": accuracy}
