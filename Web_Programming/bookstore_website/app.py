from flask import Flask, render_template, request, redirect, url_for, make_response
import mysql.connector


# instantiate the app
app = Flask(__name__)

# Connect to the database
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "20011016",
    database ="books"
)

# Retrieve all the categories from the database
cursor = mydb.cursor()
cursor.execute("SELECT * FROM category")
# Fetch all the rows in a list of tuples called categories
categories = cursor.fetchall()

# set up routes
@app.route('/')
def home():
    # Link to the index page.  Pass the categories as a parameter
    return render_template('index.html', categories=categories)


@app.route('/category') # Store the categoryId passed as a URL parameter into a variable
def category():
    selectedCategory = request.args.get('categoryId', type=int)
    # Retrieve the books from the database that are associated with the selected categoryId
    category_query = "SELECT * FROM book WHERE categoryId = %s"
    cursor.execute(category_query, (selectedCategory,))
    books = cursor.fetchall()
    # Link to the category page, pass the selectedCategory, categories, and books as parameters
    return render_template('category.html',
                           selectedCategory=selectedCategory,
                           categories=categories,
                           books=books)



@app.errorhandler(Exception)
def handle_error(e):
    """
    Output any errors - good for debugging.
    """
    return render_template('error.html', error=e) # render the edit template


if __name__ == "__main__":
    app.run(debug=True, port=8080)
