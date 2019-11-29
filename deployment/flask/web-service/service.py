import os
#Import Flask
from flask import Flask, request
from flask_cors import CORS
from keras.preprocessing import image
from ann_loader import cargarModelo
import numpy as np

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 5000
port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model,loaded_scaler,loaded_scaler_y,loaded_labelEncoderX1,graph
loaded_model,loaded_scaler,loaded_scaler_y,loaded_labelEncoderX1, graph = cargarModelo()
#global loaded_model,loaded_scaler,loaded_labelEncoderX1,graph
#loaded_model,loaded_scaler,loaded_labelEncoderX1, graph = cargarModelo()


#Define a route
@app.route('/')
def main_page():
	return 'Modelo desplegado en la Nube!'

@app.route('/rul/', methods=['GET','POST'])
def churn():
	return 'Modelo RUL de cables!'

@app.route('/rul/hoist/', methods=['GET','POST'])
def default():
	# print (request.data)
	# print (request.args)
	# print (request.form)
	data = None
	if request.method == 'GET':
		print ("GET Method")
		data = request.args

	if request.method == 'POST':
		print ("POST Method")
		if (request.is_json):
			data = request.get_json()

	print("Data received:", data)

	# Obteniendo parametros
	Marca = data.get("Marca")
	Diametro = data.get("Diametro")
	Toneladas = data.get("Toneladas")
	Leve = data.get("Leve")
	Moderado = data.get("Moderado")
	Medio = data.get("Medio")
	Severo = data.get("Severo")

	print ("\nMarca: ",Marca,
			"\nDiametro: ", Diametro,
			"\nToneladas: ", Toneladas,
			"\nLeve: ",Leve,
			"\nModerado: ",Moderado,
			"\nMedio: ",Medio ,
	                "\nSevero: ",Severo)

	# Transformado/Escalando la data
	[Marca] = loaded_labelEncoderX1.transform([Marca])
	
	hoist = np.array([Marca,Diametro,Toneladas,Leve,Moderado,Medio,Severo])
	print("\nhoist: ", hoist)
	hoist = loaded_scaler.transform([hoist])
	print("hoist Norm: ", hoist)

	with graph.as_default():
		resultado = ""
		score = loaded_model.predict(hoist)
		score_norm =  loaded_scaler_y.inverse_transform(score)
		score_norm = score_norm.astype(int)
		score_norm_f = score_norm[0]
		
		print("\nFinal score: ", score_norm_f)
				
		return ' RUL = ' + str(score_norm_f[0]) + ' Horas'

# Run de application
app.run(host='0.0.0.0',port=port)
