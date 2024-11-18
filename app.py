from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # Importar pandas

# Criação do aplicativo Flask e configuração para procurar os templates na raiz do projeto
app = Flask(__name__, template_folder='./')  # O Flask agora vai procurar na raiz

# Caminho para o arquivo do modelo
modelo_file = 'random_forest_model.pkl'

# Carregar o modelo salvo com pickle
try:
    with open(modelo_file, 'rb') as f:
        modelo = pickle.load(f)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {modelo_file} não foi encontrado.")
except EOFError as e:
    print(f"Erro: {e}")
except pickle.UnpicklingError:
    print("Erro: O arquivo não é um arquivo pickle válido.")
except Exception as e:
    print(f"Erro desconhecido: {e}")

# Rota principal que exibe o formulário
@app.route('/')
def index():
    return render_template('index.html')  # Agora procura no diretório raiz

# Rota para lidar com a previsão quando o formulário for enviado
@app.route('/previsao', methods=['POST'])
def previsao():
    try:
        # Pegar os valores dos campos preenchidos no formulário
        age = float(request.form['age'])
        sex_male = float(request.form['sex_male'])
        smoker_yes = float(request.form['smoker_yes'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        region_northwest = float(request.form['region_northwest'])

        # Adicionar a variável region_southeast, que deve ser 0 ou 1 (valor fixo 0 se não for preenchido)
        region_southeast = 0  # Definindo um valor fixo para region_southeast, já que você removeu do formulário

        # Criar um DataFrame com os dados de entrada, incluindo os nomes das colunas
        dados_entrada = pd.DataFrame([[age, sex_male, smoker_yes, bmi, children, region_northwest, region_southeast]],
                                     columns=['age', 'sex_male', 'smoker_yes', 'bmi', 'children', 'region_northwest', 'region_southeast'])

        # Fazer a previsão usando o modelo
        previsao_resultado = modelo.predict(dados_entrada)

        # Exibir o resultado na página de resultado
        return render_template('resultado.html', previsao=round(previsao_resultado[0], 2))

    except Exception as e:
        return f'Ocorreu um erro: {e}'

# Rodar o aplicativo Flask
if __name__ == '__main__':
    app.run(debug=True)
