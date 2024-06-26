# myperceptron3.py

# python -m pip install numpy
# pip install numpy
import numpy as np

class MyPerceptron:
    # -------------------------------------
    def __init__(
        self, # a instância que está a ser inicializada
        p_num_iterations = 1000, # quantidade de oportunidades para aprender
        p_learning_rate = 0.001
    ):
        self.mNumIterations = p_num_iterations
        self.mLearningRate = p_learning_rate

        # prefixo _ é uma notação sugerida para identificadores "privados"
        self.mActivationFunction = self._stepFunc
        self.mWeights = None
        self.mBias = None
    # def __init__

    #-------------------------------------
    def _stepFunc(
        self,
        # p_left_half_output
        p_linear_output # é o número q resulta do momento de ponderação linear
    ):
        # assumindo que o output é um escalar
        # return 1 if p_linear_output>=0 else 0

        # sendo genéricos para realidades mais complexas
        return np.where(
            p_linear_output>=0,
            1,
            0
        )
    # _stepFunc
    # -------------------------------------
    # processo iterativo de descoberta
    # de pesos e bias (é um processo de optimização)
    def fit(
        self,
        p_dataset,
        p_correct_corresponding_labels
    ):
        how_many_samples, nfeatures =\
            np.shape(p_dataset)

        self.mWeights = np.zeros(
            nfeatures
        )
        self.mBias = 0

        # cobrir o num de iterações previstas
        for iteration in range(self.mNumIterations):
            # o p_dataset é um iterável
            # podemos escrever ciclos como
            # for coisa in p_dataset:
            # fazer algo
            for idx, current_sample in enumerate(p_dataset):

                linear_output = np.dot(
                    current_sample,
                    self.mWeights # pesos dinâmicos
                )

                linear_output += self.mBias # bias dinâmico
                # logits são os outputs lineares sem passarem por função de ativação

                # right-half of the computing unit
                prediction = self._stepFunc(
                    linear_output
                )

                correct_classification = \
                    p_correct_corresponding_labels[idx]

                deviation = correct_classification - prediction

                adjust = self.mLearningRate * deviation

                # ajustar o quê? as coisas dinâmicas
                # isto é: self.mWeights e self.mBias
                # uma vez que current_sample é um
                # array numpy
                # isto é uma multiplicação pela técnica
                # de "Broadcasting" em que, em paralelo,
                # a multiplição de adjust faz-se por todas as features
                self.mWeights += adjust * current_sample

                self.mBias += adjust
            # for todas as amostras
        # for todas as oportunidades de aprendizagem
    # def fit

    #------------------------------------------------
    def predict(
        self,
        p_new_thing_for_classification
    ):
        linear_output = np.dot(
            p_new_thing_for_classification,
            self.mWeights # should only be called AFTER fit
        )
        linear_output += self.mBias

        prediction = self.mActivationFunction(
            linear_output
        )

        return prediction
    # def predict
# class MyPerceptron

import ferramentas
#ferramentas.visualizacao_massiva_de_dataset()

from ferramentas import visualizacao_massiva_de_dataset, \
    my_accuracy

# from ferramentas import *

# your P code
P = MyPerceptron()

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

loading_result = load_breast_cancer()
X = loading_result.data
feature_names = loading_result.feature_names

"""
visualizacao_massiva_de_dataset(
    p_dataset=X,
    p_nomes_das_features=feature_names
)
"""

quantas_amostras, quantas_features_por_amostra =\
    np.shape(X)
print(f"#Amostras: {quantas_amostras}")
print(f"#Features per sample: {quantas_features_por_amostra}")
uma_amostra_ao_calhas = np.random.randint(0, quantas_amostras)
print(feature_names)
print(X[uma_amostra_ao_calhas])

y = loading_result.target

X_train, X_test, y_train, y_test = train_test_split(
    X, # col (nº amostras, nº features) - todas as features (todas as amostras, vetorizadas)
    y, # col (nº amostras,) - todas as suas classificações corretas
    test_size=0.9, # 20 % de amostras quero reservar para aferir a qualidade do treino, qu
    random_state=123 # comparabilidade
)

P.fit(
    p_dataset=X_train,
    p_correct_corresponding_labels=y_train
)

predictions = P.predict(X_test)
print(f"Prediction: {predictions[:10]}")
print(f"Correct Classifications: {y_test[:10]}")

score =\
    accuracy_score(
        y_test,
        predictions
    )

ac = my_accuracy(predictions, y_test)

print(f"accuracy by sklearn.metrics = {score}")
print(f"accuracy by my_accuracy = {ac}")