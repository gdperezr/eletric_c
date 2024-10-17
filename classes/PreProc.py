import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from eda.eda import df
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from random import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

class PreProcess:
    def __init__(self, df):
        self.df = df

    def fill_na(self):

        """ Substituindo valores Nulos , mediana para numericos e moda para categoricos"""
        # Preenchimento para colunas numéricas
        for column in self.df.select_dtypes(include=[float, int]).columns:
            median = self.df[column].median()
            self.df[column].fillna(median, inplace=True)

        # Preenchimento para colunas categóricas
        for column in self.df.select_dtypes(include=[object]).columns:
            mode = self.df[column].mode()[0]
            self.df[column].fillna(mode, inplace=True)

        # Retornar contagem de valores ausentes após o preenchimento
        nulos = self.df.isna().sum()
        return nulos

    def train_test_split(self):
        """ Separar em treino e teste """
        seed = 42
        np.random.seed(seed)

        # Separar em Treino e Teste
        X = self.df.drop(columns=['Charging Cost (USD)'])
        y = self.df['Charging Cost (USD)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

        return X_train, X_test, y_train, y_test

    def column_transformer(self, X_train, X_test):
        """ Aplicar ColumnTransformer para codificação e escalonamento """
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        numeric_features = X_train.select_dtypes(include=[float, int]).columns

        # Configuração do ColumnTransformer com OneHotEncoder e StandardScaler
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features),
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='drop'
        )

        # Aplicar o pré-processamento nos dados de treino e teste
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        return X_train_transformed, X_test_transformed

    def model_random_forest(self):
        """ Criação do Pipeline para o modelo 1 - RandomForestRegressor """

        # Pipeline de pré-processamento e modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=seed))
        ])

        # Parâmetros para GridSearchCV
        param_grid = {
            'model__n_estimators': [100, 200,300],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }

        # Configuração e treinamento com GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Melhor modelo
        best_model = grid_search.best_estimator_
        print("Melhores parâmetros encontrados:", grid_search.best_params_)

        # Previsões no conjunto de teste
        y_pred = best_model.predict(X_test)

        # Calcular métricas de avaliação
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Resultados no conjunto de teste:")
        print(f"Erro Quadrático Médio (MSE): {mse}")
        print(f"Erro Médio Absoluto (MAE): {mae}")
        print(f"Coeficiente de Determinação (R²): {r2}")

    def model_xgboost(self, X_train, y_train, X_test, y_test):
        pipeline = Pipeline(steps=[
            ('model', XGBRegressor(random_state=42))
        ])

        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.7, 0.8, 1]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print("Melhores parâmetros encontrados:", grid_search.best_params_)

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Resultados no conjunto de teste com XGBoost:")
        print(f"Erro Quadrático Médio (MSE): {mse}")
        print(f"Erro Médio Absoluto (MAE): {mae}")
        print(f"Coeficiente de Determinação (R²): {r2}")

        return y_pred












