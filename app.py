#Import Libraries
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler


 
#import model # load model.py
 
app = Flask(__name__)
model = pickle.load(open('model_house_price_prediction.pkl', 'rb'))
@app.route('/',methods=['GET'])
 
# render htmp page
@app.route('/')
def home():
    return render_template('index.html')
 
# get user input and the predict the output and return to user
@app.route('/predict',methods=['POST'])
def predict():
     
    #take data from form and store in each feature    

        YearBuilt = int(request.form['YearBuilt'])
        GarageArea=float(request.form['GarageArea'])
        TotRmsAbvGrd=int(request.form['TotRmsAbvGrd'])
    
        OverallQua=int(request.form['OverallQua'])
        GrLivArea=int(request.form['GrLivArea'])
        FullBath=float(request.form['FullBath'])
     
    # predict the price of house by calling model.py
        a = np.array([YearBuilt,GarageArea,TotRmsAbvGrd,OverallQua,GrLivArea,FullBath])
        a = np.reshape(a,(-1,1))
        predicted_price = model.predict(a) 
 
 
    # render the html page and show the output
        return render_template('index.html', prediction_text='Predicted Price House is {}'.format(predicted_price))
 
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port="8080")
     
if __name__ == "__main__":
    app.run(debug=True)
