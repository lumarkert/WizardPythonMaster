from training_setup import TrainingSetup
from training_wrapper import TrainingWrapper


def start_training_wrapper(training_setup: TrainingSetup):
    tw = TrainingWrapper(training_setup)
    tw.start_training()

start_training_wrapper(TrainingSetup("No Decay LR Batchsize 1024 Less Games More Trains 10k Steps"))
start_training_wrapper(TrainingSetup("No Decay LR Ohe Input"))
#start_training_wrapper(TrainingSetup("Only Batchsize 1024"))
#start_training_wrapper(TrainingSetup("No Decay LR"))



