from eda.eda import df
from classes.PreProc import PreProcess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#### Pre Processamento ####


pre = PreProcess(df)


## Substituindo valores Nulos

df = pre.fill_na()

if not pre.df['car_age_group'].mode().empty:
    mode_value = pre.df['car_age_group'].mode()[0]
else:
    mode_value = '0-5'

pre.df['car_age_group'].fillna(mode_value, inplace=True)

nulos = pre.df.isna().sum()
print(nulos)


# Separando treino e teste
# X_train, X_test, y_train, y_test = pre.train_test_split()
#
# # Aplicando o modelo 1
# # df_random_forest = pre.model_random_forest()
#rm -rf .git

# pre.model_xgboost(X_train, y_train, X_test, y_test)


# Remover colunas especificadas antes do pré-processamento ( Se mostraram inuteis para o Modelo)
df = df.drop(columns=['Vehicle Age (years)', 'Distance Driven (since last charge) (km)',
                                        'Charging Rate (kW)', 'Energy Consumed (kWh)','car_age_group','User Type',
                      'Charging Station Location','Battery Capacity (kWh)','Charging Rate (kW)','temperature',
                      'Day of Week'])

# Separar em treino e teste
X_train, X_test, y_train, y_test = pre.train_test_split()

# Aplicar o ColumnTransformer
X_train_transformed, X_test_transformed = pre.column_transformer(X_train, X_test)

# Treinar e avaliar o modelo XGBoost
y_pred = pre.model_xgboost(X_train_transformed, y_train, X_test_transformed, y_test)


# Certifique-se de que o tamanho de y_pred e X_test seja o mesmo antes de concatenar
if len(y_pred) == len(X_test):
    # Criar DataFrame para as previsões e concatenar
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Charging Cost (USD)'])
    X_test = X_test.reset_index(drop=True)  # Resetar índice para garantir o alinhamento
    df_results = pd.concat([X_test, y_pred_df], axis=1)
else:
    print("Erro: o tamanho de y_pred não coincide com X_test.")

# Adicionar 'Charging Cost (USD)' ao X_test se não estiver presente
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Adiciona 'Charging Cost (USD)' ao df_results
X_test['Charging Cost (USD)'] = y_test

# Criar DataFrame para as previsões e concatenar
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Charging Cost (USD)'])
df_results = pd.concat([X_test, y_pred_df], axis=1)

# Visualizar as colunas 'Charging Cost (USD)' e 'Predicted Charging Cost (USD)'
print(df_results[['Charging Cost (USD)', 'Predicted Charging Cost (USD)']])

# Plotar a relação entre os valores reais e previstos de 'Charging Cost (USD)'
sns.lmplot(data=df_results, x='Charging Cost (USD)', y='Predicted Charging Cost (USD)', line_kws={'color': 'red'}, aspect=1.5)
plt.title('Relação entre o Custo de Carregamento Real e Previsto')
plt.xlabel('Custo de Carregamento Real (USD)')
plt.ylabel('Custo de Carregamento Previsto (USD)')
plt.show()