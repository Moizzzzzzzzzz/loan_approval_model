# Loan Approval Prediction App

A Streamlit web application that predicts loan approval status using a machine learning model.

## 🚀 Features

- **Instant Predictions**: Real-time loan approval decisions.
- **Interactive UI**: Easy-to-use form for entering applicant details.
- **Model Insights**: View key factors influencing the decision.
- **Responsive Design**: Works on desktop and mobile.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd loan-approval
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app/app.py
    ```

## ☁️ Deployment

### Streamlit Cloud (Recommended)

1.  Push this code to a GitHub repository.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Connect your GitHub account.
4.  Select your repository (`loan-approval`).
5.  Set the **Main file path** to `app/app.py`.
6.  Click **Deploy**!

### Render

1.  Push code to GitHub.
2.  Go to [render.com](https://render.com/).
3.  Create a new **Web Service**.
4.  Connect your GitHub repo.
5.  Set **Build Command** to `pip install -r requirements.txt`.
6.  Set **Start Command** to `streamlit run app/app.py --server.port $PORT`.
7.  Click **Create Web Service**.

## 📂 Project Structure

```
loan-approval/
├── model/              # Model artifacts (pkl files)
├── app/
│   └── app.py          # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```
