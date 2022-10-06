PriFL: Private Federated Learning for Inverting Gradient Attacks
=================================================

Introduction
-------------------------

The project named <b>PriFL</b> aims to evaluate whether the existing protection 
mechanisms satisfying local local differential privacy defend inverting gradient 
(<b>InvertGrad</b>) attacks threatening the security of federated learning. The 
attacks attempt to utilize the gradient update information of local neural 
models to reconstruct the training example.


Framework of federated learning:
* <b>FedAvg</b>.

Inverting gradient attacks.
* [1]

Protection approaches satisfying local differential privacy.

Neural networks:
* Multiple Layer P (MLP).
* (CNN).
* (ResNet). 

Datasets:
* <b>MNIST</b>.
* .

The framework of PriFL.
* <b>console</b>. Control the construction of a server and a set of clients. 
Control the data dispatch of clients.
* <b>Server</b>. Construct and aggregate the global model according to the 
local models from clients.
* <b>Client</b>. Train the local model over the local dataset.

Command:
> a

Key influence:
* \# of parameters vs. \# of features

Tricks:
* 

References
-------------------------

