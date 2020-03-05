from ImageServer import ImageServer
import numpy as np

# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]
imageDirs = ['/home/320077119/Desktop/DeepAlignmentNetwork/data/images/thermal_downscaled/']
boundingBoxFiles = ["../TrainingPrepare/boxesThermalDownscaledAll.pkl"]

datasetDir = "../data2/"

meanShape = np.load("../data/reducedMeanShape.npz")["meanShape"]
trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, None, meanShape, 200, 300000, False)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 200, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)