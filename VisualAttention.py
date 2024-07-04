import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report
from RNeVA import RNevaWrapper
from torch.utils.data import Subset
import numpy as np

torch.autograd.set_detect_anomaly(True)
torch.autograd.detect_anomaly()

rnn_hidden_size = 128
image_size = 224
foveation_sigma = 0.15
blur_filter_size = 5
blur_sigma = 5
forgetting = 0.3
batch_size = 64
print(torch.cuda.is_available())
# Definire il dataset e i dataloader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #nn so se è necessario
])
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
train_dataset1 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
cacca,popo = train_dataset1[0]
print(popo)

print(train_dataset1.classes)
num_samples = 2000
indices = np.random.permutation(len(test_dataset))[:num_samples]

subset = Subset(test_dataset, indices)
test_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

subset_indices = range(10000)  # Cambia 1000 con il numero desiderato di campioni
train_dataset = Subset(train_dataset1, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


downstream_model = models.resnet18(pretrained=True) 


num_features = downstream_model.fc.in_features
downstream_model.fc = torch.nn.Linear(num_features, 10)

downstream_model = downstream_model.to('cuda')
criterion = torch.nn.CrossEntropyLoss()
for param in downstream_model.parameters():
    param.requires_grad = False
def target_function(x, y):
    return y


model = RNevaWrapper(downstream_model, criterion, target_function, image_size, foveation_sigma, blur_filter_size, blur_sigma, forgetting, rnn_hidden_size).to('cuda')
optimizer = optim.Adam(list(model.rnn.parameters()) + list(model.fc.parameters()), lr=0.001)


def train(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            #print("immagine")
            #print(images.shape)
            images, labels = images.to('cuda'), labels.to('cuda')
            _, loss = model.run_optimization(images, labels, optimizer, scanpath_length=10)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.mean().item()}")
        evaluate(model,test_loader,criterion)

# Funzione per valutare il modello
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            _,outputs = model.create_scanpath(images,scanpath_length=10)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=classes, digits=4)

    print(f"Test Loss: {total_loss/len(test_loader)}, Accuracy: {100 * correct / len(test_loader.dataset)}%")
    print(accuracy)
    print(f"Classification Report:\n{report}")

train(model, train_loader, optimizer, criterion, num_epochs=10)


evaluate(model, test_loader, criterion)

torch.save(model.state_dict(), 'rneva_model.pth')

