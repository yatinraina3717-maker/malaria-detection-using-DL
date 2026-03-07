from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
MODEL_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ['Parasitized', 'Uninfected']

print("\n" + "="*70)
print("🔬 MALARIA DETECTION - LOADING MODEL")
print("="*70)

try:
    import tensorflow as tf
    from PIL import Image
    print("✓ TensorFlow imported")
except Exception as e:
    MODEL_ERROR = f"Import failed: {e}"
    print(f"X {MODEL_ERROR}")

if MODEL_ERROR is None:
    current_dir = os.getcwd()
    print(f"✓ Directory: {current_dir}")

    try:
        h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]
        if h5_files:
            MODEL_PATH = os.path.join(current_dir, h5_files[0])
            print(f"✓ Found: {h5_files[0]}")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print(f"✓ Loaded! Input: {model.input_shape}, Output: {model.output_shape}")
        else:
            MODEL_ERROR = "No .h5 file found"
            print(f"X {MODEL_ERROR}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"X {MODEL_ERROR}")

def prepare_image(image_file):
    if model is None:
        return None
    try:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        input_shape = model.input_shape
        
        if len(input_shape) == 2:
            img = img.resize((64, 64))
            img_array = np.array(img, dtype='float32') / 255.0
            img_gray = np.mean(img_array, axis=2)
            expected_size = input_shape[1]
            img_flat = img_gray.flatten()
            if len(img_flat) < expected_size:
                img_flat = np.pad(img_flat, (0, expected_size - len(img_flat)))
            else:
                img_flat = img_flat[:expected_size]
            return np.expand_dims(img_flat, axis=0)
        else:
            img_size = (input_shape[1], input_shape[2])
            img = img.resize(img_size)
            img_array = np.array(img, dtype='float32') / 255.0
            return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Image error: {e}")
        return None

@app.route('/')
def home():
    if os.path.exists('templates/index.html'):
        return render_template('index.html')
    return f'''<!DOCTYPE html>
<html><head><title>Malaria Detection</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:200px}}
.container{{max-width:900px;margin:0 auto;background:white;border-radius:20px;padding:40px;box-shadow:0 20px 60px rgba(0,0,0,0.3)}}
h1{{color:#333;margin-bottom:30px;text-align:center}}
.status{{padding:25px;border-radius:12px;margin:25px 0;text-align:center}}
.success{{background:#d4edda;color:#155724;border:2px solid #c3e6cb}}
.error{{background:#f8d7da;color:#721c24;border:2px solid #f5c6cb}}
.icon{{font-size:3rem;margin-bottom:15px}}
</style></head><body>
<div class="container">
<h1>🔬 Malaria Detection System</h1>
<div class="status success"><div class="icon">✅</div><h2>Server Running</h2><p>Port 5000 active</p></div>
<div class="status {'error' if model is None else 'success'}">
<div class="icon">{'❌' if model is None else '✅'}</div>
<h2>Model: {'NOT LOADED' if model is None else 'LOADED'}</h2>
<p>{MODEL_ERROR if MODEL_ERROR else 'Ready!'}</p></div></div></body></html>'''

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        img_array = prepare_image(file)
        if img_array is None:
            return jsonify({'success': False, 'error': 'Image processing failed'}), 500

        # Predict
        prediction = model.predict(img_array, verbose=0)
        print(f"Prediction: {prediction}")
        print(f"Shape: {prediction.shape}")

        # Get the single value
        prob = float(prediction[0][0])
        print(f"Probability value: {prob}")

        # Determine class based on threshold
        if prob > 0.5:
            pred_class = 1 # Uninfected
            confidence = prob * 100
        else:
            pred_class = 0 # Parasitized
            confidence = (1 - prob) * 100

        result = CLASS_NAMES[pred_class]
        parasitized_prob = (1 - prob) * 100
        uninfected_prob = prob * 100

        print(f"Result: {result}, Confidence: {confidence:.2f}%")

        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(confidence, 2),
            'details': {
                'parasitized_probability': round(parasitized_prob, 2),
                'uninfected_probability': round(uninfected_prob, 2)
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("\n" + "="*70)
    print(f"{'✅' if model else '❌'} Model: {'LOADED' if model else 'NOT LOADED'}")
    if model:
        print(f"    Input: {model.input_shape}, Output: {model.output_shape}")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)