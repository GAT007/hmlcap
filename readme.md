#HML 7651 : Learning and Forgetting in Text Classification
#By: Manya Jain, Amith Tallanki, Suchitra Yelchuri
📘 HMLCAP — Massed vs Interleaved Learning in Text Classification

This project explores how training schedules affect the learning dynamics of a text classifier.
We compare:

Massed practice — model sees one class at a time (blocked training)

Interleaved practice — model sees mixed classes continuously (shuffled batches)

Using the AG News Classification Dataset, we analyze not just accuracy but also how feature weights evolve during training.

All code is contained in main.ipynb.

📂 Dataset

We use the AG News dataset, downloaded with kagglehub.

The four categories are:

Label	Category
0	World
1	Sports
2	Business
3	Sci/Tech
🧹 Preprocessing

Text is prepared by concatenating each article’s title and description.
Vectorization uses TF–IDF with:

50,000 max features

Unigrams + bigrams

English stopword removal

🧠 Training Methods

We train an SGDClassifier (logistic regression using SGD) in two regimes:

Massed Practice

Data is split by class

Each epoch loops through class 0 → class 1 → class 2 → class 3

Leads to heavy oscillations in model behavior

Interleaved Practice

Batches are drawn from a fully shuffled dataset

Produces stable, smooth learning curves

Both methods run for 5 epochs with mini-batches of 256.

🔍 Feature Tracking

We track two types of features throughout training:

1. Diagnostic Words

Representative words strongly associated with each class:

government (World)

soccer (Sports)

stock (Business)

space (Sci/Tech)

2. Lexical Sets

Groups of words representing each topic:

World (e.g., “president”, “minister”, “war”)

Sports (e.g., “game”, “team”, “coach”)

Business (e.g., “market”, “shares”, “profit”)

Sci/Tech (e.g., “research”, “software”, “space”)

Common words (e.g., “news”, “said”)

We compute the average coefficient value over each set to study how feature weights stabilize.

📊 Key Results
1. Accuracy

Interleaved training reaches high accuracy quickly

Massed practice shows strong oscillations and slower stabilization

2. Diagnostic Word Coefficients

Massed practice: large coefficient swings during each class block

Interleaving: smooth, stable convergence

3. Lexical Set Coefficients

Massed: repeated rises and crashes across epochs

Interleaved: consistent, monotonic stabilization

These behaviors mirror known findings in cognitive science:
interleaving promotes more stable and generalizable learning.

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/GAT007/hmlcap.git
cd hmlcap

2. Install dependencies

If you create a requirements.txt, use:

pip install -r requirements.txt

3. Run the notebook
jupyter notebook main.ipynb

🛠️ Tools & Libraries

Python

Scikit-learn

Pandas

NumPy

Matplotlib

KaggleHub

🙌 Acknowledgments

AG News dataset

Scikit-learn team

Research on interleaved vs massed practice
