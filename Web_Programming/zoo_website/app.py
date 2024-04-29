
from flask import Flask, render_template, request, redirect, url_for, make_response
import mysql.connector
import datetime

# instantiate the app
app = Flask(__name__)

# Connect to the database
# todo: Insert code to connect to zoo database
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "20011016",
    database ="zoo"
)

# set up the routes
@app.route('/')
def home():
    # todo: Link to the index page.
    return render_template('index.html')

@app.route('/schedule')
def schedule():
    # todo: Get today's date
    d = datetime.datetime.now()
    date_string = f'{d.month}/{d.date}/{d.year}'
    # todo: Create dictionary of events
    events = {"10:00":"Zoo opens", "12:00": "Lunch Time", "15:00" : "Zoo closes"}
    # todo: Link to the schedule page.  Pass the date as a parameter
    return render_template('schedule.html', date = date_string, schedule = events)


@app.route('/animals')
def animals():
    # todo: Execute query to get the animals from the database
    cursor = mydb.cursor()
    query = "select animalType, animalSize, animalColor from animal"
    cursor.execute(query)
    # todo: Fetch all the rows in a list of tuples called animals.
    animals = cursor.fetchall()
    # todo: Link to the animals page.  Pass the animals as a parameter
    return render_template('animals.html', animals = animals)

@app.route('/newAnimal')
def newAnimal():
    cursor = mydb.cursor()
    query = "SELECT DISTINCT animalSize from animal"
    cursor.execute(query)
    sizes = cursor.fetchall()
    return render_template('newAnimal.html',sizes = sizes)

@app.route('/newAnimal', methods= ['POST'])
def insertAnimal():
    cursor = mydb.cursor()
    # Retrieve data from database
    type = request.form['animalType']
    size = request.form['animalSize']
    color = request.form['animalColor']

    # insert the new animal into the database
    query = "INSERT INTO animal (animalType, animalSize, animalColor) VALUES (%s, %s, %s)"
    record = (type, size, color)
    cursor.execute(query, record)
    mydb.commit()
    file = request.files["animalImage"]
    filename = type + ".png"
    file.save("static/images/" + filename)
    return redirect(url_for("animals"))

if __name__ == "__main__":
    app.run(debug = True)
