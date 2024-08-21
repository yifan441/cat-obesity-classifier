from dataclasses import dataclass


@dataclass
class config:
    # preprocessing
    INPUT_DIM = 224 #256
    # training
    LOG = False
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    LOG_INTERVAL = 10
    PRINT_INTERVAL = 10 
