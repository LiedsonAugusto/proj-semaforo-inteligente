from gerenciador_csv import GerenciadorCsv
from scipy import stats
import pandas as pd
import numpy as np
import os

def DF_to_JSON(df: pd.DataFrame):
    resultado = {}
    for index, data in df.iterrows():
        if not isinstance(data['OrdemVeiculos'], str):
            print(type(data['OrdemVeiculos']))
        else:
            ordemVeiculo = data['OrdemVeiculos']
            tempo = int(data['Tempo'])
            if resultado.get(ordemVeiculo) is not None:
                resultado[ordemVeiculo].append(tempo)
            else:
                resultado[ordemVeiculo] = [tempo]
    return resultado

def clear_conflit(resultado: dict):
    for veiculos, tempo in resultado.items():
        moda = stats.mode(tempo, keepdims=True)
        limiar = len(tempo)*.8
        if moda.count >= limiar:
            resultado[veiculos] = moda.mode[0]
        else:
            resultado[veiculos] = int(round(np.mean(tempo), 0))
    return resultado

def agrupar_tempos_pequenos(resultado: dict, limite=14):
    return resultado
    novo_resultado = {}
    acumulador_14 = []

    for tempo, veiculos in resultado.items():
        if tempo < limite:
            acumulador_14.append(veiculos)
        else:
            novo_resultado[tempo] = veiculos

    novo_resultado[limite].extend(acumulador_14)

    return novo_resultado

def transpose_dict(resultado: dict, void_number, ignore_minus_one=True):
    referencia = ['LIXINHO PAPAI', 'C', 'M', 'GP', 'A', 'O']
    resultado_transposto = {}

    tamanho_maximo = 0
    for ordemVeiculo in resultado.keys():
        ordemVeiculo = ordemVeiculo.split(', ')
        if ignore_minus_one:
            ordemVeiculo = [referencia.index(veiculo)/10 for veiculo in ordemVeiculo if veiculo != "-1"]
        else:
            ordemVeiculo = [referencia.index(veiculo)/10 if veiculo != "-1" else void_number/10 for veiculo in ordemVeiculo]
        if len(ordemVeiculo) > tamanho_maximo:
            tamanho_maximo = len(ordemVeiculo)

    # new_max = 0
    for ordemVeiculo, tempo in resultado.items():
        ordemVeiculo = ordemVeiculo.split(', ')
        if ignore_minus_one:
            ordemVeiculo = [referencia.index(veiculo)/10 for veiculo in ordemVeiculo if veiculo != "-1"]
        else:
            ordemVeiculo = [referencia.index(veiculo)/10 if veiculo != "-1" else void_number/10 for veiculo in ordemVeiculo]

        if len(ordemVeiculo) < tamanho_maximo:
            ordemVeiculo += [void_number] * (tamanho_maximo - len(ordemVeiculo))
        if len(ordemVeiculo) != tamanho_maximo:
            print(ordemVeiculo)
        if resultado_transposto.get(tempo) is not None:
            resultado_transposto[tempo].append(ordemVeiculo)
        else:
            resultado_transposto[tempo] = [ordemVeiculo]

    # for tempoTeste in sorted(resultado_transposto.keys(), key=int):
    #     ordensTeste = resultado_transposto[tempoTeste]
        # print(f"tempo: {tempoTeste} quantidade: {len(ordensTeste)}")
    
    # for tempo, ordemVeiculo in resultado_transposto.items():
    #     print(len(ordemVeiculo))

    return resultado_transposto, tamanho_maximo

def train_test_split_local(data_dict, test_size=.3):
    from sklearn.model_selection import train_test_split
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for label, samples in data_dict.items():
        if len(samples) < 2:
            samples = samples + samples
        elif len(samples) > 6000:
            samples = samples[:6000]
        train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=42)
        
        train_data.extend(train_samples)
        train_labels.extend([int(label)] * len(train_samples))
        test_data.extend(test_samples)
        test_labels.extend([int(label)] * len(test_samples))

    return train_data, train_labels, test_data, test_labels

def model_train_test_dataset_init(base_dataset_name, args:dict={}):
    end_not_check = " - [ ]\r"
    end_check = " = [V]\n"

    if ".csv" in base_dataset_name or ".json" in base_dataset_name:
        raise ValueError("model_train_test_dataset_init deve ser o nome base do dataset e não o caminho para .csv. Remova o CSV ou qualquer tipo de .")

    dataset_name = base_dataset_name+".json"

    arquivos_csv = [f"{f}" for f in os.listdir(".")]
    if dataset_name in arquivos_csv:
        print("Dataset encontrado", flush=True)
        with open(dataset_name, "r") as f:
            import json
            return json.load(f)
        

    print("Dataset do modelo não encontrado...", flush=True)


    caminho_dos_dados_coletados = "./dados_csv/"
    msg = f"Concatenando dados coletados de {caminho_dos_dados_coletados}"
    print(msg, end=end_not_check, flush=True)
    gcsv = GerenciadorCsv(None, ["Rua","OrdemVeiculos","Tempo","TotalVeiculos"])
    arquivos_csv = [f"{caminho_dos_dados_coletados}{f}" for f in os.listdir(caminho_dos_dados_coletados) if f.endswith('.csv')]
    df = gcsv.juntarCsv(arquivos_csv, save=False)
    print(msg, end=end_check, flush=True)

    msg = "DF -> JSON"
    print(msg, end=end_not_check, flush=True)
    resultado = DF_to_JSON(df)
    print(msg, end=end_check, flush=True)

    msg = "Resolvendo conflitos"
    print(msg, end=end_not_check, flush=True)
    resultado = clear_conflit(resultado)
    print(msg, end=end_check, flush=True)

    msg = f"Transpondo [Ordem -> Tempos] para [Tempo -> Ordens]"
    print(msg, end=end_not_check, flush=True)
    resultado_transposto, tamanho_do_input = transpose_dict(resultado, 0, ignore_minus_one=False)
    print(msg, end=end_check, flush=True)

    tempo_minimo = args.get("tempo minimo", 14)
    msg = f"Agrupando tempos < {tempo_minimo} para {tempo_minimo}"
    print(msg, end=end_not_check, flush=True)
    resultado_transposto = agrupar_tempos_pequenos(resultado_transposto, tempo_minimo)
    print(msg, end=end_check, flush=True)

    test_size = args.get("test size", .3)
    n_batches = args.get("numero de batches", 70)
    msg = f"Separando dataset: {(1-test_size)*100}% de treino e {(test_size)*100}% de teste"
    print(msg, end=end_not_check, flush=True)
    train_data, train_labels, test_data, test_labels = train_test_split_local(resultado_transposto, test_size)
    print(msg, end=end_check, flush=True)

    final_json = {
        "tempo minimo": tempo_minimo,
        "tamanho do input": tamanho_do_input,
        "tolerancia": args.get("tolerancia", 1),
        "tamanho do output": int(max(train_labels)+1),
        "next size ratio": args.get("next size ratio", .3),
        "first layer multiply": args.get("first layer multiply", 2),
        "weight decay": args.get("weight decay", .0001),
        "learning rate": args.get("learning rate", .001),
        "test size": test_size,
        "numero de batches": n_batches,
        "train data": train_data,
        "train labels": train_labels,
        "test data": test_data,
        "test labels": test_labels
    }

    with open(dataset_name, 'w') as f:
        import json
        json.dump(final_json, f, indent=2)

    return final_json

# nao me entupi