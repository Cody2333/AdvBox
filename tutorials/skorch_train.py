import torch
from torch import nn
import torch.nn.functional as F

from sklearn.externals import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from skorch import NeuralNetClassifier

mnist = joblib.load('mnist.joblib')

print('start')

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X /= 255.0

XCnn = X.reshape(-1, 1, 28, 28)

XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

print(XCnn.shape)

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))

print(mnist_dim, hidden_dim, output_dim)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out.log_softmax(dim=1)


net = NeuralNetClassifier(
    Net,
    max_epochs=15,
    lr=1,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adadelta,
    device=device,
)

net.fit(XCnn_train, y_train)

pred = net.predict(XCnn_test)

acc = np.mean(pred == y_test)
print(acc)