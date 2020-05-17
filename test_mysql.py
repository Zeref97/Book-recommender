import mysql.connector


class StoreValue(object):

  def __init__(self):
    self._user_id = []
    self._book_id = []

  @property
  def user_id(self):
    return self._user_id

  @user_id.setter
  def user_id(self, value):
    self._user_id = value

  @property
  def book_id(self):
    return self._book_id
  
  @book_id.setter
  def book_id(self, value):
    self._book_id = value


# def connect_database():
#     mydb = mysql.connector.connect(
#         host="localhost",
#         user="recommender",
#         passwd="password"
#     )
    
#     mycursor = mydb.cursor()

#     mycursor.execute("USE book_recommender")
#     return mycursor

def get_ratings():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM ratings")
    return mycursor.fetchall()

def get_books():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM books")
    return mycursor.fetchall()

def get_users():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM users")
    return mycursor.fetchall()



ratings = get_ratings()
books = get_books()
users = get_users()

id_users_books = StoreValue()

for x in ratings:
    id_users_books._user_id.append(x[0])
    id_users_books._book_id.append(x[1])

# print(id_users_books._user_id)