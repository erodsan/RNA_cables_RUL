# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.models import load_model
from sklearn.externals import joblib

def cargarModelo():

    FILENAME_MODEL_TO_LOAD = "model_cable_hoist_f.h5"
    FILENAME_SCALER_TO_LOAD = "mmScaler_f.save"
    #FILENAME_SCALER_TO_LOAD_y = "mmScaler_y_f.save"
    FILENAME_LABELENCODER_X1_TO_LOAD = "lab_enc_marca_f.save"
    MODEL_PATH = "../../../models/cables"

    # Cargar la RNA desde disco
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("Modelo cargado de disco << ", loaded_model)

    # Cargar los parametros usados
    loaded_scaler = joblib.load(MODEL_PATH + "/" + FILENAME_SCALER_TO_LOAD)
    #loaded_scaler_y = joblib.load(MODEL_PATH + "/" + FILENAME_SCALER_TO_LOAD_y)
    loaded_labelEncoderX1 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X1_TO_LOAD)

    graph = tf.get_default_graph()
    return loaded_model,loaded_scaler,loaded_labelEncoderX1,graph
