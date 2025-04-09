from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)
# Manually provided aquifer list (raw, messy)
raw_aquifers = [
    ' Alluvium', 'ALLUVIUM', 'Aeolian Alluvium (Silt/ Sand) & Hard Formation', 'Alluvium',
    'Alluvium /Lime stone', 'Alluvium(S)', 'Alluvium/Limetone', 'Alluvium/Quartzite',
    'Alluvium/Sandstone', 'Alluvium/Slate', 'BASALT', 'BGC', 'Baishakhi Formation', 'Basalt',
    'Collapsed', 'Confined', 'Confined I', 'Confined II', 'Confined III', 'Dolomite', 'Gneiss',
    'Gneisses', 'Gneisses ', 'Gr.Gneiss', 'Grainite & Gneiss', 'Granite', 'Granitic Gneiss',
    'Hills', 'Jaisalmer formation', 'Lathi Formation', 'Limestone', 'Limestone/Quartzite',
    'Mica Schist', 'Mica-schist', 'Migmetic, Granetic Gneiss', 'OLDER ALLUVIUM', 'Older Alluvium',
    'Older-Alluvium', 'Phreatic', 'Phreatic/Alluvium', 'Phreatic/Basalt', 'Phreatic/Granite',
    'Phreatic/Limestone', 'Phreatic/Sandstone', 'Phreatic/Shale', 'Phyllite', 'Phyllite/Schist',
    'Pre-monsoonatic/Basalt', 'Qtz\x02boulder', 'Qtz-boulder', 'Quartz-Mica Schist', 'Quartzite',
    'Quartzite Schist', 'SANDSTONE', 'Sandstone', 'Schist', 'Schist/Phyllite', 'Semi Confined',
    'Semi confined', 'Shale', 'Shale & Sandstone', 'Shale/Sandstone', 'Tertiary Sandstone',
    'Unconfined', 'Unconfined Aquifer', 'Unconfined Aquifier', 'Unconfined and Confined',
    'Unconfined to Semi Confined', 'Unconfined, Confined & Leaky Confined', 'Younger Alluvium',
    'confined I', 'limestone', 'sandstone', 'shale', '\xa0Sandstone'
]

# Clean and standardize
def clean_name(name):
    name = str(name).strip().lower().replace('\xa0', ' ').replace('\x02', '')
    name = name.replace('aquifier', 'aquifer')
    return name

# Clean and get unique values
cleaned_aquifers = sorted(set([clean_name(aq) for aq in raw_aquifers]))
# Optional: Capitalize for display
aquifer_list = [aq.title() for aq in cleaned_aquifers]

# Load models and encoder
model_pre = joblib.load("model_pre_monsoon.pkl")
model_post = joblib.load("model_post_monsoon.pkl")
le = joblib.load("aquifer_encoder.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            lat = float(request.form['latitude'])
            lon = float(request.form['longitude'])
            depth = float(request.form['depth'])
            aquifer = request.form['aquifer']
            pre2021 = float(request.form['pre2021'])
            post2021 = float(request.form['post2021'])

            # Encode aquifer
            aquifer_code = le.transform([aquifer])[0] if aquifer in le.classes_ else 0
            input_data = np.array([[lat, lon, depth, aquifer_code, post2021, pre2021]])

            predicted_pre = model_pre.predict(input_data)[0]
            predicted_post = model_post.predict(input_data)[0]

            prediction = {
                "pre": round(predicted_pre, 2),
                "post": round(predicted_post, 2)
            }

        except Exception as e:
            prediction = {"error": str(e)}

    return render_template("index.html", prediction=prediction, aquifers=aquifer_list)


if __name__ == '__main__':
    app.run(debug=True)
