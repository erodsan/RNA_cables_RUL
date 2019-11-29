# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.models import load_model
from sklearn.externals import joblib

def cargarModelo():

    FILENAME_MODEL_TO_LOAD = "model_tf_rn.h5"
    FILENAME_SCALER_TO_LOAD = "stdScaler.save"
    FILENAME_LABELENCODER_X1_TO_LOAD = "lab_enc_al.save"
    FILENAME_LABELENCODER_X2_TO_LOAD = "lab_enc_ns.save"
    FILENAME_LABELENCODER_X3_TO_LOAD = "lab_enc_ec.save"
    FILENAME_LABELENCODER_X4_TO_LOAD = "lab_enc_ne.save"
    FILENAME_LABELENCODER_X5_TO_LOAD = "lab_enc_vs.save"
    FILENAME_LABELENCODER_X6_TO_LOAD = "lab_enc_mr.save"
    FILENAME_LABELENCODER_X7_TO_LOAD = "lab_enc_tv.save"
    MODEL_PATH = "../../../models/hotel"

    # Cargar la RNA desde disco
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("Modelo cargado de disco << ", loaded_model)

    # Cargar los parametros usados
    loaded_scaler = joblib.load(MODEL_PATH + "/" + FILENAME_SCALER_TO_LOAD)
    loaded_labelEncoderX1 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X1_TO_LOAD)
    loaded_labelEncoderX2 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X2_TO_LOAD)
    loaded_labelEncoderX3 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X3_TO_LOAD)
    loaded_labelEncoderX4 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X4_TO_LOAD)
    loaded_labelEncoderX5 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X5_TO_LOAD)
    loaded_labelEncoderX6 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X6_TO_LOAD)
    loaded_labelEncoderX7 = joblib.load(MODEL_PATH + "/" + FILENAME_LABELENCODER_X7_TO_LOAD)

    graph = tf.get_default_graph()
    return loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2,loaded_labelEncoderX3,loaded_labelEncoderX4,loaded_labelEncoderX5,loaded_labelEncoderX6,loaded_labelEncoderX7,graph
