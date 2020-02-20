import sys
sys.path.append("/home/320077119/Desktop/DeepAlignmentNetwork/")
sys.path.append("/home/320077119/Desktop/DeepAlignmentNetwork/DeepAlignmentNetwork/")

from DeepAlignmentNetwork.ImageServer import ImageServer
from DeepAlignmentNetwork.FaceAlignmentTraining import FaceAlignmentTraining

datasetDir = "/home/320077119/Desktop/DeepAlignmentNetwork/data2/"

trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=24850_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=200_perturbations=[]_size=[112, 112].npz")

#The parameters to the FaceAlignmentTraining constructor are: number of stages and indices of stages that will be trained
#first stage training only
# training = FaceAlignmentTraining(1, [0])
#second stage training only
training = FaceAlignmentTraining(2, [0, 1])

training.loadData(trainSet, validationSet)
training.initializeNetwork()

training.train(0.0001, num_epochs=10)
