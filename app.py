import os
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configure Gemini API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# --- ROUTE: Home ---
@app.route("/")
def home():
    return render_template("index.html")


# --- ROUTE: Generate Questions ---
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        job_description = data.get("job_description", "")

        prompt = f"""
        You are an expert HR interviewer.
        Generate exactly 4 interview questions:
        - 2 general HR questions
        - 2 job-specific questions based on the following job description.

        Job Description:
        {job_description}

        Return only the questions in a numbered list format:
        1. ...
        2. ...
        3. ...
        4. ...
        """

        response = model.generate_content(prompt)
        questions_text = response.text.strip()

        questions = []
        for line in questions_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and ('.' in line[:4]):
                parts = line.split('.', 1)
                if len(parts) > 1:
                    question = parts[1].strip()
                else:
                    question = line
            else:
                question = line
            questions.append(question)

        return jsonify({"questions": questions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- ROUTE: Evaluate Answer ---
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        question = data.get("question", "")
        answer = data.get("answer", "")

        if not question or not answer:
            return jsonify({"error": "Missing question or answer"}), 400

        prompt = f"""
        You are an expert interview evaluator.
        Evaluate the candidate's answer using the following 10 parameters.
        Each parameter should have a score from 1–10.
        Also calculate a total (sum out of 100).
        Provide a 2-3 sentence summary and 3-5 improvement tips.

        Parameters:
        1. Clarity
        2. Relevance
        3. Communication
        4. Confidence
        5. Structure
        6. Technical Depth
        7. Example Quality
        8. Conciseness
        9. Authenticity
        10. Impact

        Format response as pure JSON:
        {{
          "scores": {{
            "Clarity": 0,
            "Relevance": 0,
            "Communication": 0,
            "Confidence": 0,
            "Structure": 0,
            "Technical Depth": 0,
            "Example Quality": 0,
            "Conciseness": 0,
            "Authenticity": 0,
            "Impact": 0
          }},
          "total": 0,
          "summary": "short 2-3 sentence summary",
          "improvement_tips": [
            "tip 1",
            "tip 2",
            "tip 3"
          ]
        }}

        Question: {question}
        Answer: {answer}
        """

        response = model.generate_content(prompt)

        # --- Extract text safely ---
        raw_text = ""
        if hasattr(response, "text"):
            raw_text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                raw_text = parts[0].text

        # --- Clean + parse JSON safely ---
        cleaned = raw_text.strip().replace("```json", "").replace("```", "")
        json_part = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]

        evaluation = None
        try:
            evaluation = json.loads(json_part)
        except Exception:
            evaluation = {"raw_evaluation": raw_text}

        return jsonify({"evaluation": evaluation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=True)
