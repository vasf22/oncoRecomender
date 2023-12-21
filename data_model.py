import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
import joblib

# Geração de dados fictícios para refletir variáveis do câncer de mama
data, labels = make_classification(
    n_samples=100000,
    n_features=7,
    n_informative=6,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)

# Criando DataFrame
df = pd.DataFrame(data, columns=['idade', 'tamanho_tumor', 'status_hormonal',
                                 'estadiamento', 'receptor_estrogenio', 'receptor_progesterona', 'her2_neu'])
df['Tratamento_Recomendado'] = labels

# Lógica para definir novas variáveis
df['idade'] = np.random.randint(1, 70, size=len(df))
df['tamanho_tumor'] = np.random.uniform(0.1, 10.0, size=len(df)).round(1)
df['status_hormonal'] = np.random.choice(['positivo', 'negativo'], size=len(df))
df['estadiamento'] = [f"{i//25000 + 1}" for i in range(len(df))]
df['receptor_estrogenio'] = np.random.choice(['positivo', 'negativo'], size=len(df))
df['receptor_progesterona'] = np.random.choice(['positivo', 'negativo'], size=len(df))
df['her2_neu'] = np.random.choice(['positivo', 'negativo'], size=len(df))

# Lógica para definir o tratamento recomendado
def definir_tratamento(row):
    if row['receptor_estrogenio'] == 'positivo' and row['her2_neu'] == 'positivo':
        return 1  # Tratamento recomendado
    else:
        return 0  # Sem tratamento recomendado

# Aplicar a lógica para definir o tratamento
df['Tratamento_Recomendado'] = df.apply(definir_tratamento, axis=1)

# Salvar a base de dados antes da codificação
df.to_csv('dados_clinicos_realistas.csv', index=False)

# Lidar com variáveis categóricas usando OneHotEncoder
categorical_columns = ['status_hormonal', 'estadiamento', 'receptor_estrogenio', 'receptor_progesterona', 'her2_neu']

# Criar um objeto OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Ajustar e transformar as colunas categóricas
encoded_columns = encoder.fit_transform(df[categorical_columns])

# Criar DataFrame com as colunas codificadas
df_encoded = pd.concat([df.drop(categorical_columns, axis=1), pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)

# Reorganizar as colunas para colocar 'Tratamento_Recomendado' no final
column_order = ['idade', 'tamanho_tumor'] + list(encoder.get_feature_names_out(categorical_columns)) + ['Tratamento_Recomendado']
df_encoded = df_encoded[column_order]

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('Tratamento_Recomendado', axis=1))
df_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns[:-1])
df_scaled['Tratamento_Recomendado'] = df_encoded['Tratamento_Recomendado']

# Aplicar balanceamento de classe usando Random Oversampling apenas nas classes minoritárias
ros = RandomOverSampler(sampling_strategy=0.8, random_state=42)
X_resampled, y_resampled = ros.fit_resample(df_scaled.drop('Tratamento_Recomendado', axis=1), df_scaled['Tratamento_Recomendado'])

# Verificar se o conjunto resultante tem mais de uma classe
if len(np.unique(y_resampled)) > 1:
    df_resampled = pd.DataFrame(X_resampled, columns=df_scaled.columns[:-1])
    df_resampled['Tratamento_Recomendado'] = y_resampled

    # Calcular a proporção das classes usando o conjunto resample
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
    sampling_strategy = {cls: weight for cls, weight in zip(np.unique(y_resampled), class_weights)}

    # Salvar em um arquivo CSV
    df_resampled.to_csv('dados_clinicos_realistas_resampled.csv', index=False)

    # Salvar o scaler para uso futuro
    joblib.dump(scaler, 'scaler.pkl')

    # Salvar o encoder para uso futuro
    joblib.dump(encoder, 'encoder.pkl')

    # Dividir em características (X) e alvos (y)
    X = df_resampled.drop('Tratamento_Recomendado', axis=1)
    y = df_resampled['Tratamento_Recomendado']

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ajustar a proporção de tratamento e não tratamento durante o treinamento
    ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # Calcular pesos das classes
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_resampled), class_weights)}

    # Inicializar diferentes modelos com hiperparâmetros ajustados e termos de regularização
    rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight=class_weight_dict)
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=50, learning_rate=0.05, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='sqrt')
    et_model = ExtraTreesClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', class_weight=class_weight_dict)

    # Criar um ensemble usando a técnica de voto majoritário
    ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)], voting='soft')

    # Avaliar o modelo usando validação cruzada
    cv_scores = cross_val_score(ensemble_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f'Acurácia média na validação cruzada: {np.mean(cv_scores)}')

    # Treinar o modelo
    ensemble_model.fit(X_resampled, y_resampled)

    # Avaliar o desempenho do ensemble no conjunto de teste
    accuracy = ensemble_model.score(X_test, y_test)
    print(f'Acurácia do Ensemble no Conjunto de Teste: {accuracy}')

    # Salvar o modelo treinado
    joblib.dump(ensemble_model, 'ensemble_model.pkl')
else:
    print("A estratégia de oversampling resultou em apenas uma classe. Ajuste a estratégia conforme necessário.")
