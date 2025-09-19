# 1️⃣ Base image with Python
FROM python:3.12-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy the rest of the app files
COPY . .

# 5️⃣ Expose port 8501 for Streamlit
EXPOSE 8501

# 6️⃣ Run Streamlit in headless mode
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
