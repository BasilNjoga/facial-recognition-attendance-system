from flask import Flask, render_template, flash
from flask import request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, BooleanField, ValidationError
from wtforms.validators import DataRequired , EqualTo, Length
from flask_sqlalchemy import SQLAlchemy
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import cv2
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Create a Flask Instance
app = Flask(__name__)
# Add Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/facialrecognition'
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
    # Do somme password stuff
    password_hash = db.Column(db.String(128))

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute!')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

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
    password_hash = PasswordField('Password', validators=[DataRequired(), EqualTo('password_hash2', message='Passwords Must Match')])
    password_hash2 = PasswordField('Confirm Password', validators=[DataRequired()])
    submit = SubmitField("Log in")

#Adding machine model functions 
#Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


#Create home page
@app.route("/")
def home():
    return render_template('index.html')

#When you click the take attendance button
@app.route("/start", methods=["GET"])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        mess = "There are no registered users, please add a user to continue"
        return render_template('index.html',mess=mess) 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('index.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

#Create name page
@app.route("/name", methods=['GET', 'POST'])
def name():
    name = None
    email = None
    form = NamerForm()
    #Validate Form
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()
        if user is None:
            user = Users(name=form.name.data, email=form.email.data)
            db.sesion.add(user)
            db.session.commit()
        name = form.name.data
        form.name.data = ''
        form.email.data = ''
        flash("Form Submitted Successfuly")
    our_users = Users.query.order_by(Users.date_added)
    return render_template('name.html',
        name = name,
        form = form,
        our_users = our_users)

@app.route("/add_user", methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==500:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names,rolls,times,l = extract_attendance()
        
    return render_template("add_user.html",totalreg=totalreg())

@app.route("/dashboard")
def dashboard():
    names,rolls,times,l = extract_attendance()
    return render_template("dashboard.html",names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)