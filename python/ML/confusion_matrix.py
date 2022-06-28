from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_preds, y_true, labels):    
  cm = confusion_matrix(y_true, y_preds, normalize="true")    
  fig, ax = plt.subplots(figsize=(6, 6))    
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)    
  disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)    
  plt.title("Normalized confusion matrix")    
  plt.show()
  
 
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)
