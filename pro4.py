import pandas as pd
def find_s(file_path):
    data = pd.read_csv(file_path)
    print("Training data:\n", data)
    hypothesis = ['?' for _ in data.columns[:-1]]
    for _, row in data.iterrows():
        if row.iloc[-1] == 'Yes':  # Changed to iloc for proper indexing
            hypothesis = [v if h == '?' or h == v else '?' for h, v in zip(hypothesis, row.iloc[:-1])]
    return hypothesis
print("The final hypothesis is:", find_s('/home/shrivatsa-k-s/Desktop/python/training_data.csv'))
