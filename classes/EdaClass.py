import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EdaClass:
    def __init__(self, csv):
        """
        Inicializa a classe EdaClass carregando um arquivo CSV em um DataFrame.
        """
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        pd.set_option('display.max_columns', None)  # Exibe todas as colunas ao imprimir o DataFrame

    def nulls(self):
        """
        Verifica e exibe valores nulos no DataFrame.
        """
        print('Verificando valores nulos no DataFrame:')
        print(self.df.isna().sum())

    def unique(self):
        """
        Verifica e exibe o número de valores únicos para cada coluna no DataFrame.
        """
        print('Verificando valores únicos no DataFrame:')
        print(self.df.nunique())

    def drop_col(self, columns):
        """
        Remove as colunas especificadas do DataFrame.

        Parâmetros:
            columns (list): Lista de colunas a serem removidas.

        Retorna:
            DataFrame atualizado sem as colunas especificadas.
        """
        self.df = self.df.drop(columns=columns, axis=1)
        return self.df

    def correlation(self):
        """
        Calcula e exibe um heatmap da matriz de correlação entre variáveis numéricas.

        Retorna:
            DataFrame de correlação entre as colunas numéricas.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        corr = self.df[numeric_cols].corr()

        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return corr

    def analise_groupby(self, group_by, analise_col, method, text):
        """
        Agrupa o DataFrame por uma coluna e realiza um cálculo agregado, exibindo o resultado como um gráfico de barras.

        Parâmetros:
            group_by (str): Coluna para agrupar.
            analise_col (str): Coluna na qual aplicar o método de agregação.
            method (str): Método de agregação (e.g., 'sum', 'mean').
            text (str): Título do gráfico.

        Retorna:
            DataFrame agrupado e ordenado.
        """
        veic_carga = getattr(self.df.groupby(group_by)[analise_col], method)()
        veic_carga = veic_carga.sort_values(ascending=False).reset_index()

        ax = sns.barplot(x=group_by, y=analise_col, data=veic_carga, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title(text)

        # Adiciona os valores em cima das barras
        for index, row in veic_carga.iterrows():
            ax.text(index, row[analise_col] + 1, round(row[analise_col], 1), color='black', ha="center")
        plt.show()

        return veic_carga

    def scatter(self, x, y, hue, title):
        """
        Gera um scatter plot para análise da relação entre duas variáveis, com diferenciação de cor por uma categoria.

        Parâmetros:
            x (str): Nome da coluna para o eixo X.
            y (str): Nome da coluna para o eixo Y.
            hue (str): Coluna categórica para diferenciação de cor.
            title (str): Título do gráfico.
        """
        sns.scatterplot(x=x, y=y, data=self.df, hue=hue)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(title=hue, loc='upper right')
        plt.show()

    def boxplot(self, x, y, hue, title):
        """
        Gera um boxplot para visualizar a distribuição de dados de acordo com variáveis categóricas.

        Parâmetros:
            x (str): Nome da coluna categórica para o eixo X.
            y (str): Nome da coluna numérica para o eixo Y.
            hue (str): Coluna categórica para diferenciação de cor.
            title (str): Título do gráfico.
        """
        sns.boxplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
