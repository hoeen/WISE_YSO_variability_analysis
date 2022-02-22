'''
This script loads data as RDB from table2 
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

# create wise_2 table
cursor.execute("DROP TABLE IF EXISTS wise_2;")
cursor.execute("""
    CREATE TABLE wise_2(
        Id INTEGER NOT NULL PRIMARY KEY, 
        source_Id CHAR(5),
        RAdeg FLOAT,
        DEdeg FLOAT,
        NW1 INTEGER,
        NW2 INTEGER,
        e_class CHAR(5),
        SD_sigma FLOAT,
        DeltaW2 FLOAT,
        FAP_lsp FLOAT,
        FAP_lin FLOAT,
        Sec_var CHAR(8),
        Stoch_var CHAR(9),
        SlopeW2 FLOAT,
        Period INTEGER,
        Frac_amp FLOAT,
        Cloud CHAR(16)
    );  
""")

# insert table1 into index and wise_1
with open('table2', 'r') as t2:
    ind = 1 # source table index
    source_list = list()
    
    # byte format of file
    labelpos = list()  
    for linum, line in enumerate(t2):
        if linum >= 9 and linum < 26 and linum != 24:
            # tuple 형식으로 각 라벨의 처음과 끝 위치를 저장
            labelpos.append((int(line[:4].lstrip()),
                                         int(line[5:8].lstrip())))
       
        if linum >= 47: # data starts here
            data=[]
            for (x,y) in labelpos:
                data.append(line[x-1:y].strip())
            data.insert(0,ind)

            # insert into wise_2

            # 값 없는 부분 NULL로 바꾸기
            for word_loc, words in enumerate(data):
                if bool(words) != True:
                    data[word_loc] = None
            print(data)
            cursor.execute("""
                INSERT INTO wise_2 VALUES(
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s
                    );
            """, data
            )
            ind += 1
            
            # save source_Id as list
            # if i == 24 or before_source != data1[1]:
            #     print('put sources!')
            #     before_source = data1[1]
            #     cursor.execute("""
            #         INSERT INTO source_index VALUES(
            #             %s, %s
            #         );
            #     """, (s, before_source)
            #     )
            #     s += 1

        # if linum > 500:
        #     break
    
# print('source_list:', source_list)

# db 적재 확인
# cursor.execute("SELECT * FROM wise_2;")
# print(cursor.fetchall())



# db commit
connection.commit()


