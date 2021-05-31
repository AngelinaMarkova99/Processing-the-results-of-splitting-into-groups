import sys
import numpy as np
import pandas as pd

# Скрипт для обработки результатов разбиения
# -----------------------------------------------------------------
categories = str(input("Введите названия категорий через \"-\": ")).split("-")
print("Проверьте правильность введённых данных, если всё верно, введите \"ДА\", иначе - \"НЕТ\"\n", categories)
if (str(input()).lower()== 'да'):
    print("Продолжение работы ...")
else:
    sys.exit("Начните заново")

names = str(input("Введите названия групп автоматов через \"-\": ")).split("-")
groups = input("Введите номера кластеров для групп автоматов через \"-\": ").split("-")

result = np.zeros(shape=[len(categories), len(names)])

f = open('./export.txt', 'r')
s = f.read()
table = s.split('\n')
f.close()

Product = []
Claster = []
for i in range(0, len(table)):
    var1, var2 = table[i].split('\t')
    Product.append(var1)
    Claster.append(var2)

for i in range(len(Product)):
    if(Claster[i] in groups):
        result[categories.index(Product[i])][groups.index(Claster[i])] +=1

# Вычисление процента об общих продаж
sum = np.zeros(len(names))
result_percent = result
for i in range(0, len(names)):
    for j in range(0, len(categories)):
        sum[i] += result[j][i]
for i in range(0, len(categories)):
    for j in range(0, len(sum)):
        result_percent[i][j] = result[i][j]/sum[j]*100

df = pd.DataFrame(np.array(result_percent), index=categories, columns=names)
print(df)
df.to_excel("result.xlsx")