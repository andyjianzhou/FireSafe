import sqlite3

class MyDatabase:
    def __init__(self, db):
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS address (id INTEGER PRIMARY KEY, address text)")
        self.conn.commit()

    def fetch(self):
        self.cur.execute("SELECT * FROM address")
        rows = self.cur.fetchall()
        return rows
    def check(self, longitude, latitude):
        self.cur.execute("SELECT * FROM address WHERE address=?", (str(longitude) + ", " + str(latitude),))
        rows = self.cur.fetchall()
        return rows

    def insert(self, longitude, latitude):
        # insert into table and remove duplicates
        self.cur.execute("INSERT INTO address VALUES (NULL, ?)", (str(longitude) + ", " + str(latitude),))
        self.conn.commit()

    def delete(self, id):
        #delete longitude and latitude
        self.cur.execute("DELETE FROM address WHERE id=?", (id,))
        self.conn.commit()
        
    def __del__(self):
        self.conn.close()