
# 🎬 Movie Recommendation System

A machine learning–powered **Movie Recommendation System** that suggests movies using two approaches:

* **Content-Based Filtering** → Recommends similar movies based on genres.
* **Collaborative Filtering** → Recommends movies based on user preferences using matrix factorization (NMF).

Built with **Python, Scikit-learn, Pandas, and Streamlit**.

---

## 📌 Features

* 🎥 Search movies and get **similar recommendations** (Content-Based).
* 👤 Get personalized recommendations by entering a **User ID** (Collaborative Filtering).
* 🖥️ Interactive **Streamlit web app** interface.

---

## 📂 Project Structure

```
movie-recommender/
│
├── data/                 # MovieLens dataset (u.data, u.item)
├── recommender.py        # Core ML logic (content-based + collaborative)
├── app.py                # Streamlit app (UI)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📊 Dataset

We use the **MovieLens 100k Dataset** from [GroupLens](https://grouplens.org/datasets/movielens/100k/).

* **u.data** → User ratings (`user_id`, `movie_id`, `rating`, `timestamp`)
* **u.item** → Movie details (`movie_id`, `title`, `genres`)

📥 Download dataset:
[MovieLens 100k](https://files.grouplens.org/datasets/movielens/ml-100k.zip)

Unzip and place `u.data` and `u.item` inside the **`data/`** folder.

---

## ⚙️ Installation & Setup

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. **Create a virtual environment (optional)**

```bash
python -m venv venv
# Activate
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. Open browser → [http://localhost:8501](http://localhost:8501)

---

## 🚀 How It Works

### 🔹 Content-Based Filtering

* Uses **TF-IDF Vectorization** on genres.
* Computes **cosine similarity** between movies.
* Recommends movies with similar genres to the selected one.

### 🔹 Collaborative Filtering

* Creates **user–movie rating matrix**.
* Applies **Non-negative Matrix Factorization (NMF)**.
* Predicts unseen ratings → recommends top N movies per user.

---

## 📖 Concepts Covered

* Data preprocessing with **Pandas & NumPy**
* **TF-IDF & Cosine Similarity** (Content-Based)
* **Matrix Factorization (NMF)** (Collaborative)
* Building an interactive UI with **Streamlit**

---

## 📸 Screenshots

**Homepage**

```
🎬 Movie Recommendation System
Choose recommendation type: [Content-Based | Collaborative]
```

**Example Output**

```
Selected Movie: Toy Story (1995)

### Recommended Movies:
- Aladdin (1992)
- The Lion King (1994)
- Beauty and the Beast (1991)
- Pocahontas (1995)
- Mulan (1998)
```

---

## 🛠️ Tech Stack

* **Python 3.8+**
* **Pandas, NumPy, Scikit-learn** (Machine Learning)
* **Streamlit** (Web App)

---

## 📌 Future Improvements

* Add **Hybrid Recommendation** (combine content + collaborative).
* Use **Deep Learning** models for recommendations.
* Deploy on **Heroku/Render** for public access.

---

## 👨‍💻 Author

Developed by \[Bhavith Medharametla] 🚀
