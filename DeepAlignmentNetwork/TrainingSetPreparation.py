from ImageServer import ImageServer
import numpy as np

# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]
imageDirs = ['../data/images/thermal_detected/']
boundingBoxFiles = ["../TrainingPrepare/boxesThermal.pkl"]

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 100, 100000, False)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)