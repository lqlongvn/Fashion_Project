import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Data_Fashion_Business.csv', encoding= 'unicode_escape')

# Thêm cột Total (Thành tiền) = Quantity * Price
df[['Total']] = df['Quantity']*df['Price']


# Tách riêng Date, Month, Year
df[['Date','Month','Year']] = df['Order Date'].str.split('-',expand=True) 

# Chuyển đổi Tháng từ dạng Text sang dạng số
df.loc[df['Month'] == 'Jan', 'Month_num'] = '01'
df.loc[df['Month'] == 'Feb', 'Month_num'] = '02'
df.loc[df['Month'] == 'Mar', 'Month_num'] = '03'
df.loc[df['Month'] == 'Apr', 'Month_num'] = '04'
df.loc[df['Month'] == 'May', 'Month_num'] = '05'
df.loc[df['Month'] == 'Jun', 'Month_num'] = '06'
df.loc[df['Month'] == 'Jul', 'Month_num'] = '07'
df.loc[df['Month'] == 'Aug', 'Month_num'] = '08'
df.loc[df['Month'] == 'Sep', 'Month_num'] = '09'
df.loc[df['Month'] == 'Oct', 'Month_num'] = '10'
df.loc[df['Month'] == 'Nov', 'Month_num'] = '11'
df.loc[df['Month'] == 'Dec', 'Month_num'] = '12'

print(df.head())

# 'Item Code' Tách ra để lấy Code, Size, Color và Num của hàng hóa
df[['Code']]=df['Item Code'].str.split('-').str[0]
df[['Size']]=df['Item Code'].str.split('-').str[1]
df[['Color']]=df['Item Code'].str.split('-').str[2]
df[['Num']]=df['Item Code'].str.split('-').str[3]

# 'Item Name' Tách ra để lấy Tên hàng hóa, đưa vào cột 'Code_name'
df[['Code_name']]=df['Item Name'].str.split('-').str[0]

# Gộp 2 cột Year và Month thành cột Year_Month (dưới dạng String)
df['Year_Month'] = df['Year']+'-'+df['Month_num']

# Convert to datetime format
df['year_dateformat'] = pd.to_datetime(df['Year_Month'], format='%Y-%m')


print(df.head())

df_month_year = df.groupby('Year_Month').sum()
print(df_month_year.head())
# df_month_year.to_csv('Fashion_month_year1.csv') 

# saving the dataframe 
df.to_csv('file2.csv',index=False) 
