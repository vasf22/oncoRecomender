from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load the trained ensemble model
#ensemble_model = joblib.load('ensemble_model_best_params.pkl')
ensemble_model = joblib.load('ensemble_model.pkl')

# Load the scaler and encoder
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Definir as colunas categóricas
categorical_columns = ['status_hormonal_positivo', 'estadiamento_2', 'estadiamento_3', 'estadiamento_4',
                        'receptor_estrogenio_positivo', 'receptor_progesterona_positivo', 'her2_neu_positivo']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obter os dados do formulário
        idade = float(request.form['idade'])
        tamanho_tumor = float(request.form['tamanho_tumor'])
        status_hormonal = request.form['status_hormonal']
        estadiamento = request.form['estadiamento']
        receptor_estrogenio = request.form['receptor_estrogenio']
        receptor_progesterona = request.form['receptor_progesterona']
        her2_neu = request.form['her2_neu']

        # Criar um DataFrame com os dados do usuário
        user_data = pd.DataFrame({
            'idade': [idade],
            'tamanho_tumor': [tamanho_tumor],
            'status_hormonal_positivo': [1 if status_hormonal == 'Positivo' else 0],
            'estadiamento_2': [1 if estadiamento == '2' else 0],
            'estadiamento_3': [1 if estadiamento == '3' else 0],
            'estadiamento_4': [1 if estadiamento == '4' else 0],
            'receptor_estrogenio_positivo': [1 if receptor_estrogenio == 'Positivo' else 0],
            'receptor_progesterona_positivo': [1 if receptor_progesterona == 'Positivo' else 0],
            'her2_neu_positivo': [1 if her2_neu == 'Positivo' else 0]
        })

        print("Dados do usuário:")
        print(user_data)

        # Normalizar os dados usando o mesmo scaler
        X_scaled = scaler.transform(user_data)

        print("Dados normalizados:")
        print(X_scaled)

        # Fazer a previsão usando o modelo treinado
        prediction = ensemble_model.predict(X_scaled)[0]

        print("Resultado da previsão:")
        print(prediction)

        # Mapear o resultado da previsão para uma mensagem interpretável
        resultado_tratamento = "Recomendado" if prediction == 1 else "Não Recomendado"

        return render_template('index.html', prediction=resultado_tratamento)

if __name__ == '__main__':
    app.run(debug=True)
