import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd = "root"
)

my_cursor = mydb.cursor()

my_cursor.execute("SHOW DATABASES")

for db in my_cursor:
    print(db)

"""
The following is code for creating tables from the model used;

from project_name import app, db
app.app_context().push()
db.create_all()
"""
