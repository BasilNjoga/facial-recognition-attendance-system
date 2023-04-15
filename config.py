from flask import Flask
from flask_wtf import FlaskForm
from wtfforms import StringField, SubmitField
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create a Flask Instance
app = Flask(__name__)
# Add Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/facialrecognition'
# Secret Key !
app.config['SECRET_KEY'] = "my super secret key"
# Initialize The Database
app.app_context().push()
db = SQLAlchemy(app)

#Create Model
class Myusers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

    #create A string
    def __repr__(self):
        return '<Name %r>' % self.name

@app.route("/")
def home():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)