import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/')
def helloworld():
    return 'Helloworld'
 
@app.route('/area', methods=['GET'])
@cross_origin()
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w * h)
 
@app.route('/bmi', methods=['GET'])
@cross_origin()
def bmi():
    kg = float(request.values['kg'])
    m = float(request.values['cm']) / 100.
    return str(kg / (m ** 2))


@app.route('/iris',methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['params']
    inputs = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virgina'



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
