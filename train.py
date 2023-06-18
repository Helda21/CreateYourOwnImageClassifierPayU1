import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
import argparse

class FlowerClassifier:
    def __init__(self, data_dir, gpu=False, epochs=25, arch='vgg19', learning_rate=0.001, hidden_units=4096, checkpoint=''):
        self.data_dir = data_dir
        self.gpu = gpu
        self.epochs = epochs
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.checkpoint = checkpoint

        self.device = torch.device("cuda:0" if self.gpu and torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        if self.arch == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif self.arch == 'alexnet':
            model = models.alexnet(pretrained=True)
        else:
            raise ValueError('Unexpected network architecture', self.arch)

        for param in model.parameters():
            param.requires_grad = False

        features = list(model.classifier.children())[:-1]
        num_filters = model.classifier[len(features)].in_features

        features.extend([
            nn.Dropout(),
            nn.Linear(num_filters, self.hidden_units),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(True),
            nn.Linear(self.hidden_units, len(self.image_datasets['train'].classes))
        ])

        model.classifier = nn.Sequential(*features)
        self.model = model.to(self.device)

    def train_model(self):
        dataloaders = {
            x: data.DataLoader(self.image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
            for x in self.image_datasets.keys()
        }

        dataset_sizes = {x: len(dataloaders[x].dataset) for x in self.image_datasets.keys()}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        self.model.load_state_dict(best_model_wts)

    def load_datasets(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        image_datasets = {
            x: datasets.ImageFolder(root=self.data_dir + '/' + x, transform=data_transforms[x])
            for x in data_transforms.keys()
        }

        self.image_datasets = image_datasets

    def save_checkpoint(self):
        if self.checkpoint:
            print('Saving checkpoint to:', self.checkpoint)
            checkpoint_dict = {
                'arch': self.arch,
                'class_to_idx': self.model.class_to_idx,
                'state_dict': self.model.state_dict(),
                'hidden_units': self.hidden_units
            }

            torch.save(checkpoint_dict, self.checkpoint)

    def train(self):
        self.load_datasets()
        self.load_model()
        self.train_model()
        self.save_checkpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to dataset ')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=25)
    parser.add_argument('--arch', type=str, help='Model architecture', default='vgg19')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=4096)
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file', default='')

    args = parser.parse_args()

    if args.data_dir:
        classifier = FlowerClassifier(args.data_dir, args.gpu, args.epochs, args.arch, args.learning_rate, args.hidden_units, args.checkpoint)
        classifier.train()