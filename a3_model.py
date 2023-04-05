from torch.utils.data import DataLoader, Dataset
import argparse
import torch
from torch import nn
from torch import optim
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 


class MyDataset(Dataset):
    def __init__(self, X, y) -> None:
        """ is called, when creating the object: e.g. dataset = MyDataset()
            - stores samples in instance variable self.samples
        """
        super().__init__()
        self.X=X
        self.y=y


    def __getitem__(self, idx):
        """ is called when object is accesed with squre brackets: e.g. dataset[3]
        """
        samples = list(zip(self.X, self.y))
        return samples[idx]

    def __len__(self):
        """ is called by the magic function len: e.g. len(dataset)
        """
        return len(self.y)
    
    def len_vocab(self):
        """Return the length of the vocabulary"""
        
        return len(self.X[0])


class Model(nn.Module):
    def __init__(self) -> None:
        """ is called when model is created: e.g model = Model()
            - definition of layers
        """
        super().__init__()

        self.input_layer = nn.Linear(MyDataset.len_vocab(dataset), 5)
        #self.hidden = nn.Linear(5, 5)
        self.output = nn.Linear(5, 14)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        """ is called when the model object is called: e.g. model(sample_input)
            - defines, how the input is processed with the previuosly defined layers
        """
        after_input_layer = self.input_layer(data)
        #after_hidden_layer = self.hidden(after_input_layer)
        output = self.output(after_input_layer)

        return self.softmax(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("featurefiletest", type=str, help="The file containing the table of instances and features for the test set.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.

    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    X = []
    y = []

    # reading csv file
    with open(args.featurefile, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            for string in row:
                row=string.split()
                y.append(int(row[0]))
                X.append([eval(i) for i in row[1:]])
    
    dataset = MyDataset(X,y)
    # creation of dataloader for batching and shuffling of samples
    dataloader = DataLoader(dataset,
                            batch_size=14,
                            shuffle=True,
                            collate_fn=lambda x: x)
    model = Model()
    print("Model has been called")
    # optimizer defines the method how the model is trained
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    print("Optimizer has been called")
    # the loss function calculates the 'difference' between the models output and the ground thruth
    loss_function = nn.CrossEntropyLoss()
    print("Loss function has been called")
    # number of epochs = how often the model sees the complete dataset
    print("Looking into epochs")
    for epoch in range(14):
        total_loss = 0

        # loop through batches of the dataloader
        for i, batch in enumerate(dataloader):

            # turn list of complete samples into list of inputs and list of ground_truths
            # both lists are then converted into a tensor (matrix), which can be used by PyTorch
            model_input = torch.Tensor([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])
            
            # send your batch of sentences to the forward function of the model
            output = model(model_input)

            # compare the output of the model to the ground truth to calculate the loss
            # the lower the loss, the closer the model's output is to the ground truth
            loss = loss_function(output, ground_truth.long())

            # print average loss for the epoch
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 14), end='\r')

            # train the model based on the loss:
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
        print()
    
    #TESTING PHASE
    
    #Processing the data in exactly the same manner
    X_test = []
    y_test = []
    
    with open(args.featurefiletest, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            for string in row:
                row=string.split()
                y_test.append(int(row[0]))
                X_test.append([eval(i) for i in row[1:]])               
    
    # load test data into a PyTorch Dataset object
    test_dataset = MyDataset(X_test, y_test)

    # creation of dataloader for batching and shuffling of samples
    test_dataloader = DataLoader(test_dataset,
                            batch_size=14,
                            shuffle=True,
                            collate_fn=lambda x: x)

    # set the model to evaluation mode
    model.eval()

    # iterate over test data batches and make predictions
    predicted_labels = []
    true_labels = []
    
    for batch in test_dataloader:
        # get batch inputs and true labels
        batch_inputs = torch.Tensor([sample[0] for sample in batch])
        batch_labels = torch.Tensor([sample[1] for sample in batch]).long()

        # make predictions on batch inputs
        batch_predictions = model(batch_inputs)

        # convert predictions to class labels
        batch_predicted_labels = torch.argmax(batch_predictions, dim=1)

        # Adding our predictions and the true labels to our lists
        predicted_labels.extend(batch_predicted_labels.tolist())
        true_labels.extend(batch_labels.tolist())

    # Compute accuracy
    correct_predictions = sum([1 if pred == true_label else 0 for pred, true_label in zip(predicted_labels, true_labels)])
    total_predictions = len(predicted_labels)
    test_accuracy = correct_predictions / total_predictions
    print(f"Test accuracy: {test_accuracy:.4f}")
   
    # Make the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix\n')
    print(confusion) 

    