"""
WISE YSO db access api init file

by Wooseok Park
"""
from flask import Flask, render_template
from routes import source_id_routes

app = Flask(__name__)
app.register_blueprint(source_id_routes.bp)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)
