from flask import Flask, request, jsonify, send_from_directory
import traceback
import os
import re
from flask_cors import CORS
from dotenv import load_dotenv
import joblib
import numpy as np
import requests
import json

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Input validation and sanitization functions
def sanitize_input(text):
    """Sanitize user input to prevent XSS and injection attacks"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Escape special characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
    
    return text.strip()

def validate_input(data):
    """Validate input data structure and content"""
    if not data:
        return False, "No data provided"
    
    # Check for required fields if needed
    # Add specific validation rules here
    
    return True, "Valid input"

# Initialize Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"  # Use smaller, faster model first, fallback to llama3:latest

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            print(f"✅ Ollama connected successfully. Available models: {available_models}")
            
            # Set the best available model (prefer smaller, faster models)
            global OLLAMA_MODEL
            if "llama3.2:1b" in available_models:
                OLLAMA_MODEL = "llama3.2:1b"
                print(f"🚀 Using fast model: {OLLAMA_MODEL}")
            elif "llama3:latest" in available_models:
                OLLAMA_MODEL = "llama3:latest"
                print(f"🐌 Using standard model: {OLLAMA_MODEL}")
            else:
                OLLAMA_MODEL = available_models[0] if available_models else "llama3:latest"
                print(f"📝 Using available model: {OLLAMA_MODEL}")
            
            return True
        else:
            print(f"❌ Ollama connection failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

# Test Ollama connection on startup
test_ollama_connection()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Serve the main HTML file"""
    if request.method == 'POST':
        # Handle any POST requests to root - redirect to appropriate handler
        return jsonify({"status": "error", "message": "Invalid endpoint for POST request"}), 400
    return send_from_directory('.', 'index.html')


@app.route('/ollama-predictions.html')
def ollama_predictions():
    """Serve the Ollama predictions page"""
    return send_from_directory('.', 'ollama-predictions.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    try:
        return send_from_directory('.', filename)
    except:
        # If file not found, try to serve from subdirectories
        if '/' in filename:
            return send_from_directory('.', filename)
        return "File not found", 404


@app.route('/api/ollama-status')
def get_ollama_status():
    """Check Ollama connection status"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                'status': 'connected',
                'base_url': OLLAMA_BASE_URL,
                'current_model': OLLAMA_MODEL,
                'available_models': [model['name'] for model in models]
            })
        else:
            return jsonify({'status': 'error', 'message': 'Ollama not responding'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/firebase-config')
def get_firebase_config():
    """Secure endpoint to provide Firebase configuration to client"""
    return jsonify({
        'apiKey': os.environ.get('FIREBASE_API_KEY'),
        'authDomain': os.environ.get('FIREBASE_AUTH_DOMAIN'),
        'projectId': os.environ.get('FIREBASE_PROJECT_ID'),
        'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET'),
        'messagingSenderId': os.environ.get('FIREBASE_MESSAGING_SENDER_ID'),
        'appId': os.environ.get('FIREBASE_APP_ID'),
        'measurementId': os.environ.get('FIREBASE_MEASUREMENT_ID')
    })


@app.route('/predict', methods=['POST'])
def predict_crop():
    """Handle crop recommendation prediction"""
    try:
        # Import required libraries for crop prediction
        import joblib
        import numpy as np
        
        # Load the model (you might need to adjust the path)
        model_path = 'Crop Recommendation/model/rf_model.pkl'
        encoder_path = 'Crop Recommendation/model/label_encoder.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return jsonify({"status": "error", "message": "Model files not found"}), 500
            
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        # Get form data
        data = [
            float(request.form.get('N', 0)),
            float(request.form.get('P', 0)),
            float(request.form.get('K', 0)),
            float(request.form.get('temperature', 0)),
            float(request.form.get('humidity', 0)),
            float(request.form.get('ph', 0)),
            float(request.form.get('rainfall', 0))
        ]
        
        # Make prediction
        prediction_num = model.predict([data])[0]
        prediction_label = label_encoder.inverse_transform([prediction_num])[0]
        
        return jsonify({
            "status": "success", 
            "crop": prediction_label,
            "message": f"Recommended crop: {prediction_label}",
            "input_params": {
                "N": data[0],
                "P": data[1], 
                "K": data[2],
                "temperature": data[3],
                "humidity": data[4],
                "ph": data[5],
                "rainfall": data[6]
            }
        }), 200
        
    except Exception as e:
        print(f"Crop prediction error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/download_report', methods=['POST'])
def download_report():
    """Handle PDF report download requests"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from io import BytesIO
        import datetime
        from flask import send_file
        
        # Get form data
        crop = request.form.get('crop', 'Unknown')
        
        # Create PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Header
        p.setFont('Helvetica-Bold', 18)
        p.drawString(50, height - 60, "AgriTech Analysis Report")
        
        # Date
        p.setFont('Helvetica', 10)
        p.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Result
        p.setFont('Helvetica-Bold', 14)
        p.drawString(50, height - 120, "Analysis Result:")
        p.setFont('Helvetica', 12)
        p.drawString(70, height - 140, f"Recommendation: {crop}")
        
        # Input parameters if available
        y_pos = height - 180
        p.setFont('Helvetica-Bold', 12)
        p.drawString(50, y_pos, "Input Parameters:")
        y_pos -= 20
        
        p.setFont('Helvetica', 10)
        for key, value in request.form.items():
            if key != 'crop':
                p.drawString(70, y_pos, f"{key}: {value}")
                y_pos -= 15
        
        p.showPage()
        p.save()
        buffer.seek(0)
        
        return send_file(
            buffer, 
            as_attachment=True, 
            download_name="agritech_report.pdf", 
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({"status": "error", "message": "Failed to generate PDF report"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with Ollama Llama3"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"status": "error", "message": "No message provided"}), 400
            
        user_message = sanitize_input(data['message'])
        print(f"Debug: User message: {user_message}")
        
        if len(user_message) > 1000:
            return jsonify({"status": "error", "message": "Message too long"}), 400
        
        # Create agricultural assistant prompt
        system_prompt = """You are an expert agricultural assistant named AgriBot. 
        Provide detailed, accurate and helpful responses about farming, crops, weather impact, 
        soil health, pest control, and sustainable agriculture practices. Format your answers 
        with clear concise minimal paragraphs. If asked about something outside agriculture 
        except greetings, politely decline and refocus on farming topics."""
        
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAgriBot:"
        
        # Call Ollama API with optimized parameters
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 300,  # Limit response length for faster generation
                "num_ctx": 2048,     # Reduce context window for speed
                "repeat_penalty": 1.1
            }
        }
        
        print(f"Debug: Calling Ollama with model: {OLLAMA_MODEL}")
        
        # Try with current model, fallback to llama3:latest if timeout
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_payload,
                timeout=60  # 1 minute timeout for first try
            )
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout with {OLLAMA_MODEL}, trying with llama3:latest...")
            # Fallback to llama3:latest with shorter response
            ollama_payload["model"] = "llama3:latest"
            ollama_payload["options"]["num_predict"] = 200  # Even shorter response
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_payload,
                timeout=120  # 2 minutes for fallback
            )
        
        if response.status_code == 200:
            ollama_response = response.json()
            ai_message = ollama_response.get('response', 'Sorry, I could not generate a response.')
            
            return jsonify({
                "status": "success",
                "message": ai_message,
                "model": OLLAMA_MODEL
            }), 200
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            return jsonify({"status": "error", "message": "AI service temporarily unavailable"}), 500
        
    except requests.exceptions.Timeout:
        print("Ollama request timeout")
        return jsonify({"status": "error", "message": "AI service timeout - please try again"}), 500
    except requests.exceptions.ConnectionError:
        print("Ollama connection error - is Ollama running?")
        return jsonify({"status": "error", "message": "AI service unavailable - please ensure Ollama is running"}), 500
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Chat service temporarily unavailable"}), 500


@app.route('/crop-yield-prediction')
def crop_yield_prediction():
    """Serve the crop yield prediction input page"""
    return send_from_directory('.', 'crop-yield-input.html')


@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    """Handle crop yield prediction"""
    try:
        import joblib
        import numpy as np
        import xgboost  # Explicitly import xgboost
        
        # Load models (adjust paths as needed)
        model_path = 'Crop Yield Prediction/crop_yield_app/models/xgb_crop_model.pkl'
        crop_encoder_path = 'Crop Yield Prediction/crop_yield_app/models/Crop_encoder.pkl'
        season_encoder_path = 'Crop Yield Prediction/crop_yield_app/models/Season_encoder.pkl'
        state_encoder_path = 'Crop Yield Prediction/crop_yield_app/models/State_encoder.pkl'
        
        if not all(os.path.exists(path) for path in [model_path, crop_encoder_path, season_encoder_path, state_encoder_path]):
            return jsonify({"status": "error", "message": "Model files not found"}), 500
            
        model = joblib.load(model_path)
        crop_encoder = joblib.load(crop_encoder_path)
        season_encoder = joblib.load(season_encoder_path)
        state_encoder = joblib.load(state_encoder_path)
        
        # Get form data
        data = request.get_json() if request.is_json else request.form
        
        crop = data.get('crop', '').strip()
        year = int(data.get('year', 2024))
        season = data.get('season', '').strip()
        state = data.get('state', '').strip()
        area = float(data.get('area', 0))
        production = float(data.get('production', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Validate inputs
        if not all([crop, season, state]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
            
        # Check if values exist in encoders
        if crop not in crop_encoder.classes_:
            return jsonify({"status": "error", "message": f"Unknown crop: {crop}"}), 400
        if season not in season_encoder.classes_:
            return jsonify({"status": "error", "message": f"Unknown season: {season}"}), 400
        if state not in state_encoder.classes_:
            return jsonify({"status": "error", "message": f"Unknown state: {state}"}), 400
        
        # Encode categorical variables
        crop_encoded = crop_encoder.transform([crop])[0]
        season_encoded = season_encoder.transform([season])[0]
        state_encoded = state_encoder.transform([state])[0]
        
        # Prepare features
        features = np.array([[crop_encoded, year, season_encoded, state_encoded, area, rainfall, production]])
        
        # Make prediction
        prediction = float(round(model.predict(features)[0], 2))
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "crop": crop,
            "season": season,
            "state": state,
            "input_params": {
                "crop": crop,
                "year": year,
                "season": season,
                "state": state,
                "area": area,
                "production": production,
                "rainfall": rainfall
            }
        }), 200
        
    except Exception as e:
        print(f"Crop yield prediction error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/process-loan', methods=['POST'])
def process_loan():
    try:
        json_data = request.get_json(force=True)
        
        # Validate and sanitize input
        is_valid, validation_message = validate_input(json_data)
        if not is_valid:
            return jsonify({"status": "error", "message": validation_message}), 400
        
        # Sanitize any text fields in the JSON data
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if isinstance(value, str):
                    json_data[key] = sanitize_input(value)
        
        print(f"Received JSON: {json_data}")

        prompt = f"""
You are a financial loan eligibility advisor specializing in agricultural loans for farmers in India.

You will be given a JSON object that contains information about a farmer's loan application. The fields in this JSON will vary depending on the loan type (e.g., Crop Cultivation, Farm Equipment, Water Resources, Land Purchase).
You will focus only on loan schemes and eligibility criteria followed by:
1. Indian nationalized banks (e.g., SBI, Bank of Baroda)
2. Private sector Indian banks (e.g., ICICI, HDFC)
3. Regional Rural Banks (RRBs)
4. Cooperative Banks
5. NABARD & government schemes
Do not suggest generic or international financing options.

JSON Data = {json_data}

Your task is to:
1. Identify the loan type and understand which fields are important for assessing that particular loan.
2. Analyze the farmer's provided details and assess their loan eligibility.
3. Highlight areas of strength and areas where the farmer may face challenges.
4. If any critical data is missing from the JSON, point it out clearly.
5. Provide simple and actionable suggestions the farmer can follow to improve eligibility.
6. Suggest the government schemes or subsidies applicable to their loan type.
7. Ensure the tone is clear, supportive, and easy to understand for farmers.
8. Respond in a structured format with labeled sections: Loan Type, Eligibility Status, Loan Range, Improvements, Schemes.
9. **IMPORTANT: Return your response in **Markdown format** with:
Headings for each section (Loan Type, Eligibility Status, Loan Range, Improvements, Schemes)
Bullet points ( - ) for lists.
Do not use "\\n" for newlines. Instead, structure properly.

Do not add assumptions that are not supported by the data provided.
"""

        # Call Ollama API for loan processing with optimized parameters
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more consistent financial advice
                "top_p": 0.8,
                "num_predict": 500,  # Limit response length
                "num_ctx": 2048,     # Reduce context window
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=ollama_payload,
            timeout=120  # Increased timeout to 2 minutes
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
            
        ollama_response = response.json()

        reply = ollama_response.get('response', 'Unable to process loan analysis.')
        return jsonify({"status": "success", "message": reply, "model": OLLAMA_MODEL}), 200

    except Exception as e:
        print(f"Unexpected Error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)