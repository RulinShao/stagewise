from sythetic import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
num_modalities = 3
num_data = 100000
batch_size = 1000
EPOCHS = 10

# Data
train_loader, test_loader = get_data_loaders(num_modalities, num_data, batch_size)

# Start Training...
model_list = []
test_acc_list = []

# First model
modality_index = 0

model = UnimodalModel(100, 64, 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


print(f'Training unimodal model for modality {modality_index}..')
for epoch in range(EPOCHS):
    train_loss, train_acc = train_uni(model, train_loader, optimizer, criterion, [modality_index])
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

test_loss, test_acc = eval_uni(model, test_loader, criterion, [modality_index])   
print(f'\t Modality: {modality_index} | Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

model_list.append({'model': model, 'modality':[modality_index]})
test_acc_list.append(test_acc)

# Residual uni-model
for modality_index in range(1, num_modalities):

    model = UnimodalModel(100, 64, 1, res=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f'Training unimodal model for modality {modality_index}..')
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_res(model, train_loader, optimizer, criterion, [modality_index], model_list)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss, test_acc = eval_res(model, test_loader, criterion, [modality_index], model_list)    
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
    
    model_list.append({'model': model, 'modality':[modality_index]})
    test_acc_list.append(test_acc)

# Residual bi-model
intersections = get_intersections(num_modalities=num_modalities)
bi_intersections = [inter for inter in intersections if len(inter)==2 ]
for inter in bi_intersections:
    modality_index = [int(inter[0])-1, int(inter[1])-1]

    model = BimodalModel(100, 64, 1, res=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f'Training unimodal model for modality {modality_index[0]} and {modality_index[1]}..')
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_res(model, train_loader, optimizer, criterion, modality_index, model_list)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss, test_acc = eval_res(model, test_loader, criterion, modality_index, model_list)    
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
    
    model_list.append({'model': model, 'modality':modality_index})
    test_acc_list.append(test_acc)

print(test_acc_list)