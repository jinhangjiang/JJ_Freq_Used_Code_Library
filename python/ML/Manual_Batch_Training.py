from tqdm import tqdm_notebook
batchsize = round(len(df) / 100)
temp_df = pd.DataFrame()

for i in tqdm_notebook(range(0, len(df), batchsize)):
    batch = df[i:i+batchsize]

    '''
    process and train the data here
    
    output = model.train(batch)
    '''
    
    temp_df = temp_df.append(output)
