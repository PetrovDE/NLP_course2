from flask import Flask, render_template, request, jsonify
from corssencode_inference import get_range_answers, get_best_answer

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('get_answer') == 'One answer':
            one_answer = get_best_answer(request.form.get('context'), request.form.get('question'))

            return jsonify(
                {
                    "response_code": "200",
                    "request": f"{request.form.get('context')} [Cont_token] {request.form.get('question')}",
                    "response": one_answer
                }
            )
        elif request.form.get('get_answer_corpus') == 'Five answer':
            many_answer = get_range_answers(request.form.get('context'), request.form.get('question'))

            return jsonify(
                {
                    "response_code": "200",
                    "request": f"{request.form.get('context')} [Cont_token] {request.form.get('question')}",
                    "response": many_answer
                }
            )
    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run('localhost', 5000)