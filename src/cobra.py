import numpy as np
from .denoise import denoiseMethods
from .helper import *

def cobraModelInit(trainNames, noiseType, imShape, patchSize=1, best=True):

    trainingData, trainingData1, trainingData2, testingData = loadTrainingData(
        trainNames, noiseType, patchSize)
    denoisemethods = denoiseMethods()
    epsilon = 0.2
    machines = 4
    cobra = Cobra(epsilon=epsilon, machines=machines)
    cobra.fit(trainingData, testingData)

    for i, denoise in enumerate(denoisemethods):
        cobra.load_machine(denoise, CobraMachine(denoise, i, patchSize))

    cobra.load_machine_predictions()
    print("Predictions:," cobra.machine_predictions_)

    if best:

        cobra_diagnostics = Diagnostics(cobra, trainingData, testingData)
        epsilon, _ = cobra_diagnostics.optimal_epsilon(
            trainingData, testingData, line_points=100, info=False)
        machines, _ = cobra_diagnostics.optimal_alpha(
            trainingData, testingData, epsilon=Epsilon_opt, info=False)

        cobra = Cobra(epsilon=epsilon, machines=machines)
        cobra.fit(
            trainingData,
            testingData,
            default=False,
            X_k=trainingData1,
            X_l=trainingData2,
            y_k=testingData,
            y_l=testingData)
        for i, denoise in enumerate(denoisemethods):
            cobra.load_machine(denoise, CobraMachine(denoise, i, patchSize))
        cobra.load_machine_predictions()
        print("Predictions:," cobra.machine_predictions_)

    return cobra, alpha, epsilon