from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create a Flask Instance
app = Flask(__name__)
# Add Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/facialrecognition'
# Secret Key !
app.config['SECRET_KEY'] = "my super secret key only i know"
# Initialize The Database
db = SQLAlchemy(app)

#Create Model
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

    #create A string
    def __repr__(self): 
        return '<Name %r>' % self.name

# Create a Form Class
class UserForm(FlaskForm):
    name = StringField("Name:", validators=[DataRequired()])
    email = StringField("Email:", validators=[DataRequired()])
    
    submit = SubmitField("Add User")
class NamerForm(FlaskForm):
    name = StringField("Name:", validators=[DataRequired()])
    email = StringField("Email:", validators=[DataRequired()])
    
    submit = SubmitField("Log in")


#Create home page
@app.route("/")
def home():
    return render_template('index.html')

#Create name page
@app.route("/name", methods=['GET', 'POST'])
def name():
    name = None
    email = None
    form = NamerForm()
    #Validate Form
    if form.validate_on_submit():
        name = form.name.data
        form.name.data = ''
        flash("Form Submitted Successfuly")

    return render_template('name.html',
        name = name,
        form = form)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    name = None
    form = UserForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()
        if user is None:
            user = Users(name=form.name.data, email=form.email.data)
            db.session.add(user)
            db.session.commit()
        name = form.name.data
        form.name.data = ''
        form.email.data = ''
        flash(" has been added")
    our_users = Users.query.order_by(Users.date_added)
    return render_template("add_user.html",
        form = form,
        name = name,
        our_users=our_users)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)