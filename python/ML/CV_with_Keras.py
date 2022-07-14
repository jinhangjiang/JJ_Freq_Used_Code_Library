from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
rmse_nn = []
r2_nn = []
smape_nn = []
mae_nn = []

for train_index, test_index in tqdm_notebook(kfold.split(X)):

  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = df.TobinsQ[train_index], df.TobinsQ[test_index]

  scaler = StandardScaler()
  X_train_scale = scaler.fit_transform(X_train)
  X_test_scale = scaler.fit_transform(X_test)
  # Define and fit the model
  %%time
  text_input = layers.Input(shape=(len(X_train[0]),), dtype='float32', name='text')
  out_dense = layers.Dense(768, activation='PReLU')(text_input)
  out_dense1 = layers.Dense(100, activation='relu')(out_dense)
  out = layers.Dense(1, activation='linear')(out_dense1)
  # At model instantiation, we specify the input and the output:
  model = Model(text_input, out)
  model.compile(optimizer='adam',
                loss='mse',
                metrics=['mape'])

  model.fit(X_train_scale, y_train, epochs=20, validation_split = 0.375, callbacks =[call_reduce])
  nn_prediction = model.predict(X_test_scale)
  def flatten(xss):
    return [x for xs in xss for x in xs]
  nn_prediction = flatten(nn_prediction)

  rmse_nn.append(mean_squared_error(y_test, nn_prediction, squared=False))
  r2_nn.append(r2_score(y_test, nn_prediction))
  smape_nn.append(smape(y_test, nn_prediction))
  mae_nn.append(mean_absolute_error(y_test, nn_prediction))

print('-'*20, 'NN Evaluation', '-'*20)
print('RMSE: ', np.mean(rmse_nn))
print('R2 Score: ', np.mean(r2_nn))
print('SMAPE: ', np.mean(smape_nn))
print('MAE: ', np.mean(mae_nn))
