batchsize = round(len(df) / 100)
temp_df = pd.DataFrame()
final_df2_len=0
for i in tqdm_notebook(range(0, len(df), batchsize)):
    batch = df_final_miss[i:i+batchsize]

    '''
    process and train the data here
    
    output = model.train(batch)
    '''
    
    temp_df = temp_df.append(output)
