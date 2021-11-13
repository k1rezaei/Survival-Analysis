import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path_to_file = 'prepared_for_rsf_KNN_imputation.csv'

a = pd.read_csv(path_to_file)

random_state = 64
df = a.sample(frac=1.0, random_state=random_state)
df = pd.get_dummies(data=df)

df['dead'] = df['dead'].replace(2, False).replace(1, True)

# y = np.array(list(zip(df['dead'].replace(2, False).replace(1, True), df['EZ'])),
#             dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

y = df[['dead', 'EZ']]
y = y.to_numpy()
y_cols = ['Status', 'Survival']

Xt = df.drop(columns=['dead', 'EZ'])
x_cols = Xt.columns


Xt = Xt.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    Xt, y, test_size=0.25, random_state=random_state)


train_x_df = pd.DataFrame(X_train, columns=x_cols)
test_x_df = pd.DataFrame(X_test, columns=x_cols)

train_y_df = pd.DataFrame(y_train, columns=y_cols)
test_y_df = pd.DataFrame(y_test, columns=y_cols)

train_x_df.to_csv('train_x.csv', index=False)
test_x_df.to_csv('test_x.csv', index=False)
train_y_df.to_csv('train_y.csv', index=False)
test_y_df.to_csv('test_y.csv', index=False)