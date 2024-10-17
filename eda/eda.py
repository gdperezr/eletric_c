import pandas as pd
from classes.EdaClass import EdaClass
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

# Carregando o DataFrame
eda = EdaClass('C:/Users/Gdper/eletric_car/.venv/csv/ev_charging_patterns.csv')
df = eda.df

###### Análise Exploratória dos Dados ######

# Resumo estatístico e estrutura do DataFrame
print(df.describe())
print(df.info())

# Verificando valores nulos
# eda.nulls()

# Verificando valores únicos
# eda.unique()

# Removendo colunas desnecessárias para a análise
df = eda.drop_col(['User ID', 'Charging Station ID', 'Charging Start Time', 'Charging End Time',])

# Verificando a correlação entre variáveis numéricas
correlation_matrix = eda.correlation()

###### Análises Agrupadas ######

# Analisando a capacidade média da bateria por modelo de veículo
veic_carga = eda.analise_groupby('Vehicle Model', 'Battery Capacity (kWh)', 'mean',
                                 'Capacidade média da bateria por modelo de carro')

# Analisando o tempo médio de carga por modelo de veículo
veic_tempo_carga = eda.analise_groupby('Vehicle Model', 'Charging Duration (hours)', 'mean',
                                       'Tempo médio de carga por modelo de veículo')

# Analisando a distância média percorrida desde a última carga por modelo de veículo
veic_distancia_ultima_carga = eda.analise_groupby('Vehicle Model', 'Distance Driven (since last charge) (km)', 'mean',
                                                  'Distância média desde a última carga por modelo de veículo')

# Analisando o custo médio de carga por modelo de veículo
veic_custo_medio_carga = eda.analise_groupby('Vehicle Model', 'Charging Cost (USD)', 'mean',
                                             'Custo médio de carga por modelo de veículo')

###### Engenharia de Atributos ######

# Criando categorias de idade do veículo
df['car_age_group'] = pd.cut(df['Vehicle Age (years)'],
                             bins=[0, 5, 10, 20, float('inf')],
                             labels=['0-5', '6-10', '11-20', 'acima de 20'])
df.drop(columns='Vehicle Age (years)', inplace=True)

# Analisando capacidade média da bateria por faixa etária do veículo
veic_age_capacidade = eda.analise_groupby('car_age_group', 'Battery Capacity (kWh)', 'mean',
                                          'Capacidade média da bateria por faixa etária do veículo')

# Analisando custo médio de carga por faixa etária do veículo
veic_age_custo = eda.analise_groupby('car_age_group', 'Charging Cost (USD)', 'mean',
                                     'Custo médio de carga por faixa etária do veículo')

# Analisando custo médio de carga por tipo de usuário
veic_user_type_custo = eda.analise_groupby('User Type', 'Charging Cost (USD)', 'mean',
                                           'Custo médio de carga por tipo de usuário')

# Analisando custo médio de carregamento por localização da estação de carregamento
carregador = eda.analise_groupby('Charging Station Location', 'Charging Cost (USD)', 'mean',
                                 'Custo médio de carregamento por localização')

# Agrupando e contando veículos por localização da estação e modelo de veículo
veiculos_mais_carregados_loc = df.groupby(['Charging Station Location', 'Vehicle Model']).size().reset_index(name='Count')
sns.barplot(x='Charging Station Location', y='Count', hue='Vehicle Model', data=veiculos_mais_carregados_loc, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Número de veículos por localização da estação de carregamento e modelo')
plt.legend(title='Vehicle Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

###### Análises Temporais e Condicionais ######

# Analisando custo médio por dia da semana
custo_medio_day_week = eda.analise_groupby('Day of Week', 'Charging Cost (USD)', 'mean',
                                           'Custo médio por dia da semana')

# Analisando custo médio por hora do dia
custo_medio_time_of_day = eda.analise_groupby('Time of Day', 'Charging Cost (USD)', 'mean',
                                              'Custo médio por hora do dia')

# Criando categorias de temperatura
df['temperature'] = pd.cut(df['Temperature (°C)'],
                           bins=[-11, 0, 10, 20, float('inf')],
                           labels=['-11-0', '0-10', '11-20', 'acima de 30'])
df.drop(columns='Temperature (°C)', inplace=True)

# Analisando custo médio de carga por categoria de temperatura
custo_medio_temperatura = eda.analise_groupby('temperature', 'Charging Cost (USD)', 'mean',
                                              'Custo médio por categoria de temperatura')

###### Análises de Dispersão ######

# Analisando a relação entre idade do veículo e capacidade da bateria
idade_capacidade = eda.scatter('car_age_group', 'Battery Capacity (kWh)', hue='car_age_group',
                               title='Idade do veículo vs Capacidade da bateria')

# Analisando a influência da temperatura no consumo de energia
temp_influence = eda.scatter('temperature', 'Energy Consumed (kWh)', 'User Type',
                             title='Temperatura vs Consumo de energia')

# Analisando a relação entre tipo de carregador e energia consumida
energy_cons_charger_type = eda.scatter('Energy Consumed (kWh)', 'Charging Cost (USD)', 'Charger Type',
                                       title='Tipo de carregador vs Energia consumida')

###### Análise de Distribuição com Boxplot ######

# Analisando a distribuição de energia consumida por tipo de carregador
boxplot_energy_consumed_charger_type = eda.boxplot(x='Charger Type', y='Energy Consumed (kWh)', hue='Charger Type',
                                                   title='Tipo de carregador vs Energia consumida')


