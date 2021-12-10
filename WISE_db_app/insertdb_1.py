'''
This script loads data as RDB from table1
from Park et al.(2021)
'''
import psycopg2
import csv
import numpy as np

# call db and insert fields
import db_connect
connection = db_connect.wise_connect().connect()
cursor = connection.cursor()
print(connection, cursor)
# create index table
# cursor.execute("DROP TABLE IF EXISTS source_index;")
cursor.execute("""
    CREATE TABLE source_index(
        Id INTEGER NOT NULL PRIMARY KEY,
        source_Id CHAR(5)
    )
""")

# create wise_1 table
# cursor.execute("DROP TABLE IF EXISTS wise_1;")
cursor.execute("""
    CREATE TABLE wise_1(
        Id INTEGER NOT NULL PRIMARY KEY, 
        source_Id CHAR(5),
        Mjd FLOAT, 
        Mag FLOAT, 
        E_mag FLOAT,
        Band CHAR(2)
    );  
""")

# insert table1 into index and wise_1
with open('table1', 'r') as t1:
    i = 1 
    ind = 1
    source_list = list()
    s = 1 # source table index
    for line in t1:
        if i > 23: # data starts here
            data1 = line.rstrip().split()
            data1.insert(0,ind)  
            print(data1)
            cursor.execute("""
                INSERT INTO wise_1 VALUES(
                    %s, %s, %s, %s, %s, %s
                    );
            """, data1
            )
            ind += 1
            
            # save source_Id as list
            if i == 24 or before_source != data1[1]:
                print('put sources!')
                before_source = data1[1]
                cursor.execute("""
                    INSERT INTO source_index VALUES(
                        %s, %s
                    );
                """, (s, before_source)
                )
                s += 1
                

        i += 1

        # if i > 80:
        #     break
    
# print('source_list:', source_list)

# db 적재 확인
# cursor.execute("SELECT * FROM wise_1;")
# print(cursor.fetchall())
# cursor.execute("SELECT * FROM source_index;")
# print(cursor.fetchall())


# db commit
connection.commit()


