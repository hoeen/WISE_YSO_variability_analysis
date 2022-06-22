"""
WISE YSO db access api init file

by Wooseok Park
"""
from flask import Flask, render_template
from db_app.routes import source_id_routes
import os

app = Flask(__name__)
app.register_blueprint(source_id_routes.bp)

# ml 모델 위치 config
app.config['ML_model'] = os.path.join(os.path.dirname(__file__), '../model/model.pkl')

# predict에 쓰일 데이터 config
app.config['wise_df'] = os.path.join(os.path.dirname(__file__), '../model/wise_df.pkl')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)
