import pandas as pd

# Знімаємо обмеження на кількість відображуваних рядків у терміналі
pd.set_option('display.max_rows', 100) 

df = pd.read_csv(r'D:\Python_Projects\ML_Labs\Lab_2\data.csv')

# Виводимо перші 100 рядків
print(df.head(100))