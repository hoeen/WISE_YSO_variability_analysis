"""
WISE YSO db access api routes

by Wooseok Park
"""
from flask import Blueprint, Response, current_app
# from flask.templating import render_template

bp = Blueprint('source_id', __name__, url_prefix='/')

# @bp.route('/')
# def index():
#     return render_template('source_id.html') 
#     #'YSO source index page 입니다. \
#     #    <br>YSO 인덱스 번호를 URL에 입력하여 정보를 다운로드할 수 있습니다.'


# 1. 소스 id 쳤을때 데이터 csv로 반환 
## 1) db 접속하여 쿼리 진행
## 2) db 결과를 받아 csv로 저장. 
@bp.route('/<catalog_id>/getcsv', methods=['GET'])
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

    # save dataframe to session
    # session['result_df'] = result_df

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

# 2. 머신러닝 모델을 이용하여 YSO를 검색하면 다음 관측에서의 등급을 
# 예측해서 알려주기. 플롯을 같이 띄우기
@bp.route('/<catalog_id>/predict')
def predict_next_point(catalog_id: str):
    import pickle, xgboost
    import matplotlib.pyplot as plt
    import matplotlib
    from flask import send_file
    from io import BytesIO
    matplotlib.use('agg')
    
    # 이전 라우트에서의 result_df 를 어떻게 넘겨받나?
    # session 을 통해 넘겨받는 방법을 이용한다.
    # result_df = session['result_df']

    # 받아온 데이터를 통해 예측값 반환

    # 받아온 데이터와 예측값 같이 plot하기. 예측값은 마지막날 + 200일

    
    with open(current_app.config['wise_df'],'rb') as pickle_file:
        data_to_predict = pickle.load(pickle_file)   
    
    
    ####### 붙여넣음 #######
    dat_pred = data_to_predict[data_to_predict['source_idx'] == catalog_id]
    column = dat_pred.columns[1:-1] # 원래 모델에 먹일 폼 생성

    # 실제로는 첫 날짜 뺌.
    dat_pred = dat_pred[dat_pred.columns[1:].drop('m1')]
    dat_pred.columns = column

    # predict using model
    model = xgboost.XGBRegressor()
    # breakpoint()
    model.load_model(current_app.config['ML_model'])
    pred = model.predict(dat_pred)
    # breakpoint()
    # plot result
    # x grid : date
    dategrid = range(0,2800,200)
    # print(len(dategrid))
    plt.plot(dategrid[:-1], 
             dat_pred[dat_pred.columns[-13:]].values[0],
             'k.',
             ms=10
    )
    # plot predicted point as red
    plt.plot(dategrid[-1],
             pred,
             'r*',
             ms=15,
             label='Predicted'

    )
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel('days')
    plt.ylabel('W2 magnitude')
    plt.title(f'Predict star brightness: {catalog_id}', size=15)
    
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    plt.close()
    ## object를 읽었기 때문에 처음으로 돌아가줌
    img.seek(0)
    return send_file(img, mimetype='image/png')
    # return str(dat_pred), 200
    ##########3
    


    # return f"""
    #    {pred} 
    # """, 200