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
#global loaded_model,loaded_scaler, graph loaded_model,loaded_scaler, graph = cargarModelo()
global loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2,loaded_labelEncoderX3,loaded_labelEncoderX4,loaded_labelEncoderX5,loaded_labelEncoderX6,loaded_labelEncoderX7, graph
loaded_model,loaded_scaler,loaded_labelEncoderX1,loaded_labelEncoderX2,loaded_labelEncoderX3,loaded_labelEncoderX4,loaded_labelEncoderX5,loaded_labelEncoderX6,loaded_labelEncoderX7, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Modelo desplegado en la Nube!'

@app.route('/hotel/', methods=['GET','POST'])
def churn():
	return 'Modelo de Segmentacion de Huespedes!'

@app.route('/hotel/huesped/', methods=['GET','POST'])
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
	Edad = data.get("Edad")
	Actividad_Laboral = data.get("Actividad_Laboral")
	Nivel_Socioeconomico = data.get("Nivel_Socioeconomico")
	Estado_Civil = data.get("Estado_Civil")
	Nivel_Educacion = data.get("Nivel_Educacion")
	Viaja_Solo = data.get("Viaja_Solo")
	Medio_Reserva = data.get("Medio_Reserva")
	Registros_Hotel = data.get("Registros_Hotel")
	Tipo_Viaje = data.get("Tipo_Viaje")

	print ("\nEdad: ",Edad,
			"\nActividad_Laboral: ", Actividad_Laboral,
			"\nNivel_Socioeconomico: ", Nivel_Socioeconomico,
			"\nEstado_Civil: ", Estado_Civil,
			"\nNivel_Educacion: ", Nivel_Educacion,
			"\nViaja_Solo: ", Viaja_Solo,
			"\nMedio_Reserva: ", Medio_Reserva,
	                "\nRegistros_Hotel: ", Registros_Hotel,
			"\nTipo_Viaje: ", Tipo_Viaje)

	# Transformado/Escalando la data
	[Actividad_Laboral] = loaded_labelEncoderX1.transform([Actividad_Laboral])
	[Nivel_Socioeconomico] = loaded_labelEncoderX2.transform([Nivel_Socioeconomico])
	[Estado_Civil] = loaded_labelEncoderX3.transform([Estado_Civil])
	[Nivel_Educacion] = loaded_labelEncoderX4.transform([Nivel_Educacion])
	[Viaja_Solo] = loaded_labelEncoderX5.transform([Viaja_Solo])
	[Medio_Reserva] = loaded_labelEncoderX6.transform([Medio_Reserva])
	[Tipo_Viaje] = loaded_labelEncoderX7.transform([Tipo_Viaje])


	huesped = np.array([Edad,Actividad_Laboral,Nivel_Socioeconomico,Estado_Civil,Nivel_Educacion,Viaja_Solo,Medio_Reserva,Registros_Hotel,Tipo_Viaje])
	print("\nhuesped: ", huesped)
	huesped = loaded_scaler.transform([huesped])
	print("huesped Norm: ", huesped)

	with graph.as_default():
		resultado = ""
		score = loaded_model.predict(huesped)
		score_norm = (score > 0.5)
		score_norm = score_norm.astype(int)
		print("\nFinal score: ", score_norm)
		
		grupo = np.argmax(score_norm) + 1

		return ' Score: ' + str(score_norm[0]) + '  -->  Grupo '+ str(grupo)

# Run de application
app.run(host='0.0.0.0',port=port)
