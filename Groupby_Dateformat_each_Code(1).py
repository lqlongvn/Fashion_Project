import pandas as pd
df = pd.read_csv('Groupby_dateformat(1)(1).csv', encoding= 'unicode_escape')
df1=df.query('Code == "BL"')
print(df1.head())
df1.to_csv('BL(1).csv', index=False)
