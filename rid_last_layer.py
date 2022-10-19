INPUT_FILE = "/home/lab530/KenYu/aicpu/con_testing.csv"
OUTPUT_FILE = "/home/lab530/KenYu/aicpu/con_testing_summit.csv"

import pandas as pd 

df = pd.read_csv(INPUT_FILE)

print(df)
df = df.drop(['last_layer'], axis=1)
print(df)



df.to_csv(OUTPUT_FILE, index = False)

