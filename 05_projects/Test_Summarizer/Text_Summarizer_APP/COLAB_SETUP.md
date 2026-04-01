# 🚀 Run Text Summarizer on Google Colab

This guide explains how to run the FastAPI-based Text Summarizer using Google Colab.

---

## 📌 Prerequisites

* Google account (for Colab + Drive)
* your Saved model & tokenizer folder (`saved_summary_model`) in Google Drive
* **Ngrok** account (for public URL access)

---

## 📂 Project Structure (Collab )

```
/content/
│
├── app.py
├── index.html
├── saved_summary_model/
```

---

## 🧠 Step-by-Step Setup

---

### ✅ Step 1: Install Required Libraries

Run this in a Colab cell:

```python
!pip install fastapi uvicorn pyngrok nest_asyncio transformers torch
```

---

### ✅ Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

👉 This allows access to your saved model.

---

### ✅ Step 3: Copy Model to Colab

```python
!cp -r /content/drive/MyDrive/saved_summary_model /content/
```

👉 Improves performance compared to loading directly from Drive.

---

### ✅ Step 4: Create FastAPI App (`app.py`)

```python
%%writefile app.py
# paste all code inside - app.py
```
paste all code of [app.py](./app.py)  
👉 This creates a Python file used to run the server.
>or refer to [app.ipynb](./app.ipynb)


---

### ✅ Step 5: Add HTML UI

```python
%%writefile templates/index.html
<!-- paste your HTML code here -->
```
paste HTML code [index.html](./index.html)

---

### ✅ Step 6: Verify

```python
!ls
# check must curr dir: 
# app.py drive index.html saved_summary_model
```

---

### ✅ Step 7: Start FastAPI Server

```python
import subprocess

process = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
)
```

👉 Runs the server in background.

---

### ✅ Step 8: Start Ngrok Tunnel

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

public_url = ngrok.connect(8000)
print(public_url)
```
For ngrok token visit: https://dashboard.ngrok.com/get-started/your-authtoken

👉 This generates a public URL.

---

### 🌐 Step 9: Open in Browser

Copy the generated URL and open it:

```
https://your-ngrok-url
```

👉 You will see the Text Summarizer UI.

---

## 🧪 Testing the API

Open:

```
https://your-ngrok-url/docs
```

👉 Access Swagger UI to test endpoints.

---

## 🧠 Notes

* Colab is temporary → files are deleted after session ends
* Always re-run all cells after restart
* Use Google Drive for persistent storage

---

🎉 Your FastAPI Text Summarizer is now live on Colab!