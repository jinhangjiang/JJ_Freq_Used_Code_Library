# smape
def compute_metrics(eval_pred):

  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)

  # SMAPE
  smape = 1/len(labels) * np.sum(2 * np.abs(predictions-labels) / (np.abs(labels) + np.abs(predictions))*100)
  # 'micro', 'macro', etc. are for multi-label classification. If you are running a binary classification, leave it as default or specify "binary" for average
  return {"smape": smape} 
