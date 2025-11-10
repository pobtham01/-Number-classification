import sqlite3

conn = sqlite3.connect('SVM.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS users (ID INTEGER, name TEXT)')
conn.commit()
conn.close()

def insert(ID, name):
    conn = sqlite3.connect('SVM.db')
    cursor = conn.cursor()
    insert_query = 'INSERT INTO users (ID, name) VALUES (?, ?)'
    cursor.execute(insert_query,(ID, name))
    conn.commit()
    conn.close()

# insert(0, '0')
# insert(1, '1')
# insert(2, '2')
# insert(3, '3')  
# insert(4, '4')
# insert(5, '5')
# insert(6, '6')
# insert(7, '7')
# insert(8, '8')
# insert(9, '9')