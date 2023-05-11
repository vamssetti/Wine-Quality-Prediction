from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("new.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    finall=np.asarray(int_features)
    final=finall.reshape(1,-1)
    print(finall)
    print(final)
    output=model.predict(final)
    print(output)
    if(output==1):
        output = "Good"
    else:
         output = "Bad"
    return render_template('new.html',pred='wine is {}'.format(output))
    
        

if __name__ == '__main__':
    # Debug/Development
    # app.run(debug=True, host="0.0.0.0", port="5000")
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
