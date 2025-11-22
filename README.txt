===========================================
AI DOCUMENT SUMMARIZER - README
===========================================

📁 DESCRIPTION:
--------------
This project is a full-stack AI-powered system for summarizing business documents (PDF), and generating downloadable summary reports.

It includes:
- A Tkinter-based Desktop Client UI
- A Flask-based AI Server with summarization, classification, and historical analysis

===============================
🔧 SETUP INSTRUCTIONS
===============================

📌 Step 1: Clone or copy the project files to your machine.

📌 Step 2: Install Python 3.10 or newer.

📌 Step 3: Create a virtual environment (optional but recommended)
--------------------------------------------------------------
For Windows:
> python -m venv venv
> venv\Scripts\activate

For Linux/macOS:
$ python3 -m venv venv
$ source venv/bin/activate

📌 Step 4: Install server dependencies
-------------------------------------
> pip install -r requirements_server.txt

📌 Step 5: Install client dependencies
-------------------------------------
> pip install -r requirements_client.txt

📌 Step 6: Run the Flask Server
-------------------------------
> python server.py

(Default port is 5000. You can change it inside the code.)

📌 Step 7: Run the Tkinter Desktop Client
----------------------------------------
> python client.py

===============================
🌐 API ENDPOINTS (Flask Server)
===============================

1. POST /upload
   - Accepts multiple PDF files
   - Returns summarized text + PDF download URL

2. GET /download/<filename>
   - Downloads the generated summary PDF

3. GET /history
   - Returns metadata of recent uploads

4. GET /relationships/<doc_id>
   - Shows similar documents and comparison

5. GET /cleanup
   - Deletes old cache and records

6. GET /stats
   - Returns document counts and server metrics

7. GET /health
   - Simple health check for server uptime

===============================
💻 REQUIREMENTS
===============================

✅ Server:
- Flask
- Hugging Face Transformers (facebook/bart-large-cnn)
- pdfplumber
- reportlab
- scikit-learn
- sqlite3
- redis
- numpy
- joblib
- opencv-python

✅ Client:
- ttkbootstrap
- PyPDF2
- python-docx
- requests
- tkinter (pre-installed with Python)

===============================
📝 SAMPLE USAGE
===============================

1. Launch the server:
   > python server.py

2. Launch the client:
   > python client.py

3. In the client app:
   - Enter server IP and port (e.g., 127.0.0.1 and 5000)
   - Select files to upload
   - Click "Upload to Server"
   - Save and preview the generated summary PDF

===============================
📞 SUPPORT
===============================
If anything breaks or behaves unexpectedly, check:
- Server logs (app.log)
- Console errors
- Firewall blocking localhost or port 5000

NOTE: The insights feature hasn't been fully developed due to the complexity in integrating the SQLite so the insight section would return an error saying "Historical analysis unavailable due to processing error"

