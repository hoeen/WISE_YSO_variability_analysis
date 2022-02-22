"""
WISE YSO db access api routes

by Wooseok Park
"""
from flask import Blueprint, Response
from flask.templating import render_template

bp = Blueprint('source_id', __name__, url_prefix='/source_id')

@bp.route('/')
def index():
    return render_template('source_id.html') 
    #'YSO source index page 입니다. \
    #    <br>YSO 인덱스 번호를 URL에 입력하여 정보를 다운로드할 수 있습니다.'


# 2. 소스 id 쳤을때 데이터 csv로 반환 
## 1) db 접속하여 쿼리 진행
## 2) db 결과를 받아 csv로 저장. 
@bp.route('/<catalog_id>', methods=['GET'])
def get_csv_by_catid(catalog_id: str):
    import db_connect
    import pandas as pd
    import io
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
    # stringIO 스트림에 df 저장
    output_stream = io.StringIO()
    result_df.to_csv(output_stream)
    response = Response(
        output_stream.getvalue(),
        mimetype='text/csv',
        content_type='application/octet-stream'
    )
    response.headers['Content-Disposition'] = \
        f'attachment; filename={catalog_id}_data.csv'

    return response, 200