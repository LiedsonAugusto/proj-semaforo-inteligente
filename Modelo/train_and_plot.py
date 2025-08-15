from model_dataset_loader import model_train_test_dataset_init
from torch.utils.data import DataLoader, TensorDataset
from custom_loss import TolerantCrossEntropyLoss
import matplotlib.pyplot as plt
import torch, random, os

def treinar(args):
    global stop_train

    train_loader = args["train loader"]
    test_loader = args["test loader"]
    len_test_data = args["len test data"]
    model = args["model"]
    loss_fn = args["loss fn"]
    optimizer = args["optimizer"]
    learning_rate = args["learning rate"]
    model_path = args["model path"]
    tolerancia = args["tolerancia"]

    plt.ion()
    fig, (ax1, ax3) = plt.subplots(nrows=2, figsize=(10, 7))
    limite_do_plot = 50
    loss_train_history = []
    loss_test_history = []
    acuracy_history = []
    x_data = [i+1 for i in range(limite_do_plot)]
    lista_de_ruido = [-.002, -.001, .001, .002]
    numero_de_loss_minimo = 0.5
    loss_medio_atual = float('inf')
    
    model.train()

    epoch = 1
    while loss_medio_atual > numero_de_loss_minimo:
        i = 0
        loss_total = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            
            data = data.float()  # entrada como float
            labels = labels.long()
            # print(labels.max(), labels.min())

            outputs = model(data)       # saída sem softmax
            loss = loss_fn(outputs, labels)  # CrossEntropy espera logits + índices

            loss.backward()
            optimizer.step()

            # print(f'Epoch {epoch}, Loss: {loss.item():.4f}', end='\r')
            loss_total += loss.item()

            if len(loss_train_history) == 0:
                loss_train_history = [loss.item()+random.choice(lista_de_ruido) for i in range(limite_do_plot)]
            loss_train_history.append(loss.item())
            loss_train_history.pop(0)
            

            i += 1
            
            ax1.cla()
            ax1.relim()
            ax1.autoscale_view()
            ax1.set_ylabel("Loss do treinamento")
            ax1.plot(x_data, loss_train_history, label='Treinamento')
        # plt.draw()
        # print()
        slope = "None"
            
        model.eval() 
        with torch.no_grad():
            j = 0
            loss_total_eval = 0.0
            for data, labels in test_loader:
                data = data.float()
                labels = labels.long()
                outputs_eval = model(data)
                loss_eval = loss_fn(outputs_eval, labels)
                loss_total_eval += loss_eval.item()

                if len(loss_test_history) == 0:
                    loss_test_history = [loss_eval.item()+random.choice(lista_de_ruido) for i in range(limite_do_plot)]
                loss_test_history.append(loss_eval.item())
                loss_test_history.pop(0)

                j += 1

            ax1.plot(x_data, loss_test_history, color='orange', label=f'Teste | Slope {slope}')
            ax1.legend()

        acertos = 0
        for data, labels in test_loader:
            previsao = torch.argmax(model(data), dim=1)
            for p, l in zip(previsao, labels):
                if l-tolerancia <= p and l+tolerancia >= p:
                    acertos += 1

        acuracia = (acertos/len_test_data)*100
        if len(acuracy_history) == 0:
            acuracy_history = [acuracia+random.choice(lista_de_ruido) for i in range(limite_do_plot)]
        acuracy_history.append(acuracia)
        acuracy_history.pop(0)
        ax3.cla()
        ax3.relim()
        ax3.autoscale_view()
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylabel(f"Acuracia\nTolerancia {tolerancia}s")
        ax3.plot(x_data, acuracy_history)
        plt.tight_layout()
        plt.pause(0.05)

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Finalizando epoch {epoch} | LR: {learning_rate} | Acurácia: {acuracia:.2f} | Loss Média: {(loss_total/i):.4f} | Loss Eval Média: {(loss_total_eval/j):.4f}')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path)
        epoch += 1






args = {
    "tempo minimo": 14,
    "test size": .3,
    "numero de batches": 50,
    "next size ratio": .05,
    "first layer multiply": 50,
    "weight decay": .0001,
    "learning rate": .001,
    "tolerancia": 1
}

dict_dataset = model_train_test_dataset_init("model_dataset", args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torch.tensor(dict_dataset["train data"], dtype=torch.float32 , device=device)
train_labels = torch.tensor(dict_dataset["train labels"], dtype=torch.long , device=device)
test_data = torch.tensor(dict_dataset["test data"], dtype=torch.float32 , device=device)
test_labels = torch.tensor(dict_dataset["test labels"], dtype=torch.long , device=device)

train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

batch_size = int(train_labels.shape[0]/dict_dataset["numero de batches"])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

from semaforo_inteligente import SemaforoInteligente2
model = SemaforoInteligente2(
            input_size=dict_dataset["tamanho do input"], 
            output_size=dict_dataset["tamanho do output"],
            next_size_ratio=dict_dataset["next size ratio"],
            first_layer_mulpl=dict_dataset["first layer multiply"]
        )
# model.load_state_dict(torch.load("modelo_bonito.pth"))
model.to(device)
print(model)
input()

train_args = {
    "train loader": train_loader,
    "test loader": test_loader,
    "len test data": len(dict_dataset["test data"]),
    "model": model,
    "loss fn": TolerantCrossEntropyLoss(tolerance=dict_dataset["tolerancia"]),
    "optimizer": torch.optim.AdamW(
        model.parameters(), 
        lr=dict_dataset["learning rate"], 
        weight_decay=dict_dataset["weight decay"]
    ),
    "learning rate": dict_dataset["learning rate"],
    "model path": "modelo_bonito.pth",
    "tolerancia": dict_dataset["tolerancia"],
}
treinar(train_args)