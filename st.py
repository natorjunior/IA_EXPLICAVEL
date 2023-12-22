import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")



variedades_imagens = {
    "setosa": "https://raw.githubusercontent.com/natorjunior/turma-319/main/fundamentos-de-machine-learning/material-de-apoio/setosa.png",
    "versicolor": "https://raw.githubusercontent.com/natorjunior/turma-319/main/fundamentos-de-machine-learning/material-de-apoio/versicolor.png",
    "virginica": "https://raw.githubusercontent.com/natorjunior/turma-319/main/fundamentos-de-machine-learning/material-de-apoio/virginica.png",
}
# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Converter rótulos para categorias binárias
y = to_categorical(y, num_classes=3)

# Divida os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['variedade'] = iris.target
dic_iris = ['setosa', 'versicolor', 'virginica']
df['variedade'] = df['variedade'].apply(lambda x: dic_iris[x])

st.title('IA Explicável (XAI)')

side_sel = st.sidebar.selectbox('Página:',['Resumo','IRIS Setosa/MLP'])

if side_sel == 'Resumo':
    st.markdown('''
Para aplicar o LIME em um modelo MLP treinado com Keras para explicar decisões sobre a classe Iris Setosa, 
vamos primeiro treinar um modelo no conhecido conjunto de dados Iris e, 
em seguida, usar o LIME para interpretar uma instância específica. 
''')
elif side_sel == 'IRIS Setosa/MLP':
    radio = st.radio('Selecione:',['IRIS', 'MLP', 'LIME', 'APP'],horizontal=True)
    if radio == 'IRIS':
        col1_,col2_ = st.columns(2)
        with col1_:
            variedade = st.selectbox('VARIEDADE:', ['setosa','versicolor','virginica'])
        with col2_:
            coluna_iris = st.selectbox('COLUNA:', df.columns[:-1])
        #    st.image(variedades_imagens[variedade],width=100)
        #variedade = st.selectbox('VARIEDADE:', ['setosa','versicolor','virginica'])
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(df.query(f'variedade == "{variedade}"'))
        with col2:
            # Crie o boxplot para cada característica numérica
            fig = px.box(df, x="variedade", y=coluna_iris, color="variedade",
             title="Distribuição do comprimento da sépala para cada espécie de Iris")
            st.plotly_chart(fig)

            

    elif radio == 'MLP':
        st.code('''import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.utils import to_categorical

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Converter rótulos para categorias binárias
y = to_categorical(y, num_classes=3)

# Divida os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie o modelo
model = Sequential()
model.add(Dense(8, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes de saída

# Compile o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treine o modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

''')
    elif radio == 'LIME':
        selecao = st.selectbox('SELECIONE:',['RESUMO','CODIGO','RESULTADOS'])
        if selecao == 'RESUMO':
            st.markdown('''
O LIME (Local Interpretable Model-agnostic Explanations) é uma técnica de interpretabilidade de modelos de machine learning que explica as previsões 
                        de qualquer classificador de uma maneira compreensível para os seres humanos. O objetivo do LIME é permitir que você entenda 
                        por que um modelo tomou uma certa decisão. Ele é chamado de "model-agnostic", o que significa que pode ser usado com qualquer modelo,
                         independentemente de sua complexidade ou tipo.
- **Localidade:** O LIME foca em entender o modelo localmente, ao redor da previsão que está sendo explicada. Em vez de tentar compreender todo o modelo (que pode ser muito complexo), ele se concentra em uma previsão específica.

- **Perturbação de Dados:** O LIME cria um novo conjunto de dados, consistindo em amostras perturbadas (modificadas) ao redor do ponto de interesse (a instância que você quer explicar). Por exemplo, se estiver explicando uma previsão de imagem, ele pode alterar alguns pixels; para dados tabulares, ele pode modificar os valores das características.

- **Previsão do Modelo:** O LIME, então, usa o modelo original para prever as saídas para essas novas amostras perturbadas.
''')
        elif selecao == 'CODIGO':
            st.code("""import lime
import lime.lime_tabular

# Inicialize o explainer LIME para dados tabulares
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification')

# Selecione uma instância do conjunto de teste da classe Iris Setosa (classe 0)
setosa_idx = np.where(np.argmax(y_test, axis=1) == 0)[0][0]

# Explicar a predição
exp = explainer.explain_instance(X_test[setosa_idx], model.predict, num_features=4)

# Mostrar a explicação
print('Instância explicada:', setosa_idx)
print('Previsão do modelo:', iris.target_names[np.argmax(model.predict(X_test[setosa_idx:setosa_idx+1]))])
exp.show_in_notebook(show_table=True)

""")
        elif selecao == 'RESULTADOS':
            pass
    elif radio == 'APP':
        sepal_length = st.sidebar.slider('sepal_length',0.0,10.0,1.0)
        sepal_width  = st.sidebar.slider('sepal_width',0.0,10.0,1.0)
        petal_length  = st.sidebar.slider('petal_length',0.0,10.0,1.0)
        petal_width  = st.sidebar.slider('petal_width',0.0,10.0,1.0)
        model = load_model('iris_model.h5',compile=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        valor = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
        #st.write(valor.shape)
        pred = model.predict(valor)
        st.write(np.argmax(pred))
        # Inicialize o explainer LIME para dados tabulares
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            mode='classification')

        # Explicar a predição
        exp = explainer.explain_instance(valor[0], model.predict, num_features=4)

        # Mostrar a explicação
        #print('Instância explicada:', pred)
        a = iris.target_names[np.argmax(pred)]
        st.write(f'Previsão do modelo: {a}')
            # Gere a figura da explicação
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)



