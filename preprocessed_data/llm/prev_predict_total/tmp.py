import pandas as pd

df = pd.read_csv('predict_total.csv')

print(len(df[df['prediction'] == '긍정']))
print(len(df[df['prediction'] == '중립']))
print(len(df[df['prediction'] == '부정']))