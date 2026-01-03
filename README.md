# 📧 Email / SMS Spam Detection System

A Machine Learning–based **Spam Detection System** that classifies messages as **Spam** or **Not Spam (Ham)**.  
The project includes model training, evaluation, and a **Streamlit web app** for real-time testing.


## 🚀 Features
- Detects **Spam vs Ham** messages
- Uses **TF-IDF vectorization**
- Trained with multiple ML models (best model saved)
- Interactive **Streamlit UI**
- Shows prediction confidence (if supported)
- Lightweight & fast inference


## 🧠 Machine Learning Models Used to select the best Model
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear)
- Random Forest
- AdaBoost
- Gradient Boosting
- Bagging Classifier
- XGBoost

> Final model was selected based on Precision and Accuracy.


## 🛠️ Tech Stack
- **Python**
- **Scikit-learn**
- **NLTK**
- **XGBoost**
- **Streamlit**
- **Pickle**
- **Pandas**
- **Numpy**



## 📂 Project Structure

```bash
spam_detection_system/
├── app.py
├── model.pkl
├── model.ipynb    
├── spam.csv        # Training Dataset
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```


## ⚙️ Text Preprocessing
The following steps are applied to input messages:
- Remove special characters
- Convert to lowercase
- Tokenization
- Stopword removal
- Stemming (Porter Stemmer)

This ensures **training and inference consistency**.


## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sayan-Mondal2022/spam-detection-system.git
cd spam-detection-system
```

### 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv .venv
```

### 3️⃣ Activate the virtual environment
```bash
source .venv\Scripts\activate
```

### 4️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Streamlit application
```bash
streamlit run app.py
```

### 6️⃣ Open in browser
```bash
http://localhost:8501
```

## 🧪 Sample Test Messages

Spam Example
```bash
URGENT! You have won ₹50,000 cash.
Click here to claim now!
```

Ham Example
```bash
Hi, I’ll be late today. Let’s talk in the evening.
```

## 🙏 Acknowledgement

I would like to express my gratitude to the open-source community and the developers of libraries such as **Scikit-learn**, **NLTK**, **pandas**, **numpy** and **Streamlit**, which made this project possible.  
Their well-documented tools and resources played a crucial role in building, training, and deploying this spam detection system.

Special thanks to online learning platforms, tutorials, and datasets that helped me understand text preprocessing, feature engineering, and model evaluation in depth.


## 🙌 Thank You

Thank you for taking the time to explore this project!  
I hope this spam detection system is helpful for learning, experimentation, and real-world understanding of text classification.

If you have any feedback, suggestions, or improvements, feel free to reach out or raise an issue.  
Happy coding! 🚀