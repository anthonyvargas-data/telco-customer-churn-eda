import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def clasificar_variables(self):
        numericas = [col for col in self.df.columns if str(self.df[col].dtype) in ['int64', 'float64'] and col != 'SeniorCitizen']
        categoricas = [col for col in self.df.columns if col not in numericas]
        return numericas, categoricas

    def obtener_descriptivas(self):
        cols_num = self.df.select_dtypes(include=['int64', 'float64']).columns
        cols_a_describir = [c for c in cols_num if c != 'SeniorCitizen']
        return self.df[cols_a_describir].describe()

    def analizar_nulos(self):
        nulos = self.df.isnull().sum()
        porcentaje = (self.df.isnull().sum() / len(self.df)) * 100
        return pd.DataFrame({'Conteo': nulos, 'Porcentaje': porcentaje}).sort_values(by='Conteo', ascending=False)

    def plot_nulos_pie(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        total_nulos = self.df.isnull().sum().sum()
        total_datos = self.df.size
        
        if total_nulos > 0:
            labels = ['Nulos', 'Completos']
            sizes = [total_nulos, total_datos - total_nulos]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=140)
            plt.title("Proporción de Valores Nulos en el Dataset")
        else:
            ax.text(0.5, 0.5, "No hay valores nulos", ha='center', va='center')
        return fig

    def analizar_categoricas(self, columna):
        conteo = self.df[columna].value_counts()
        proporcion = self.df[columna].value_counts(normalize=True)
        return pd.DataFrame({'Conteo': conteo, 'Proporción': proporcion})

    def plot_distribucion(self, columna, tipo='hist'):
        fig, ax = plt.subplots(figsize=(8, 4))
        if tipo == 'hist':
            sns.histplot(data=self.df, x=columna, kde=True, ax=ax)
        elif tipo == 'box':
            sns.boxplot(data=self.df, x=columna, ax=ax)
        return fig

    def plot_barras(self, columna):
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df[columna].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        plt.xticks(rotation=45)
        return fig

    def plot_bivariado_num_cat(self, num_col, cat_col):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=ax, palette="muted")
        return fig

    def plot_bivariado_cat_cat(self, col1, col2):
        tabla = pd.crosstab(self.df[col1], self.df[col2])
        fig, ax = plt.subplots(figsize=(8, 4))
        tabla.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        plt.xticks(rotation=45)
        return fig, tabla