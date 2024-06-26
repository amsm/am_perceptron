# ferramentas.py
import numpy as np

def visualizacao_massiva_de_dataset(
    p_dataset,
    p_nomes_das_features
):
    for index_da_amostra, amostra in enumerate(p_dataset):
        for index_da_feature, nome in enumerate(p_nomes_das_features):
            valor_da_feature_na_amostra_corrente =\
                amostra[index_da_feature]
            msg = f"Value of {nome} @{index_da_amostra} = "+\
                  f"{valor_da_feature_na_amostra_corrente}"

            b_ultima_feature = index_da_feature==len(p_nomes_das_features)-1
            if(b_ultima_feature):
                msg+="\n"
                msg+="_.-^-."*10

            print(msg)
        # for
    # for
# def visualizacao_massiva_de_dataset

def my_accuracy(
    p_predictions,
    p_real_deal
):
    comparison = p_predictions == p_real_deal
    how_many_trues = np.sum(comparison)
    total = len(p_predictions)
    accuracy = how_many_trues / total
    return accuracy
# def my_accuracy