def file_download(catalog_id: str):
    import db_connect
    import pandas as pd
    connection = db_connect.wise_connect().connect()
    cursor = connection.cursor()
    cursor.execute('''
        SELECT * FROM wise_1 w1
        WHERE w1.source_id = (%s);
    ''', (catalog_id,)
    )
    result = cursor.fetchall()
    result_df = pd.DataFrame(result, 
                columns=['index','source_id', 'mjd', 'mag', 'emag', 'band'],
               
                )
    # delete index column
    result_df = result_df.drop(columns=['index'])

    return result_df

print(file_download('M244'))


