from flask import Flask, request, render_template, jsonify , redirect


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])

def home():
    return render_template('index.html')

@app.route('/display_posetron', methods=['GET', 'POST'])

def posetron():

    return redirect("http://localhost:5000/")
    # return render_template('posetron.html')

@app.route('/display_converserator', methods=['GET', 'POST'])

def converserator():
    return render_template('converserator.html')



if __name__ == '__main__':
    app.run(debug=True)