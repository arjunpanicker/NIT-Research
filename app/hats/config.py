class CONFIG:
    # Constant values

    ## ML Model Constants
    THRESHOLD = 0.60

    ## File names
    MODEL_SAVEFILE = 'ft_model.ftz'
    EMBEDDINGS_SAVEFILE = 'embeddings.pickle'
    COMMANDS_ONLY_CSV = 'comm_only.csv'
    STOP_WORD_TXT = 'stop_words.txt'
    OUTPUT_ZIP_FILE = 'Home_Auto_Outputs.zip'
    OUTPUT_DATASET_FILE = 'comm_preprocessed.txt'
    SVM_MODEL_SAVEFILE = 'svm_model.h'
    LR_MODEL_SAVEFILE = 'lr_model.h'
    KNN_MODEL_SAVEFILE = 'knn_model.h'

    ## Feature Names
    INPUT_FEATURE = 'sentence_embedding'
    TARGET = 'encoded_label'

    # Fasttext Hyperparameters
    FT_DIMS = 150

    ## File Manager Constants
    NN_OUTPUT_DIRECTORY_NAME = 'output/nn_models/'
    OUTPUT_DIRECTORY_NAME = 'output/'
    PLOT_DIRECTORY_NAME = 'plots/'
    DATASET_DIRECTORY = 'dataset/'