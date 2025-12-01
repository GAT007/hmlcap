# HML 7651 : Learning and Forgetting in Text Classification  
### By: Manya Jain, Amith Tallanki, Suchitra Yelchuri

---

# 📘 HMLCAP — Massed vs Interleaved Learning in Text Classification

This project explores how **training schedules** affect the learning dynamics of a text classifier.  
We compare two training strategies:

- **Massed practice** — model sees one class at a time (blocked training)  
- **Interleaved practice** — model sees mixed classes continuously (shuffled batches)

Using the **AG News Classification Dataset**, we analyze not just accuracy but also how **feature weights change over time**.

All code is contained in **`main.ipynb`**.

---

## 📂 Dataset

We use the AG News dataset, downloaded with `kagglehub`.

| Label | Category |
|-------|----------|
| 0     | World    |
| 1     | Sports   |
| 2     | Business |
| 3     | Sci/Tech |

---

## 🧹 Preprocessing

Text is constructed by concatenating each article’s title and description.  
We apply **TF–IDF vectorization** with:

- 50,000 max features  
- Unigrams + bigrams  
- English stopword removal  

---

## 🧠 Training Methods

We train an `SGDClassifier` (logistic regression via stochastic gradient descent) in two regimes:

### **Massed Practice**
- Data is split by class  
- Each epoch iterates: class **0 → 1 → 2 → 3**  
- Produces large oscillations in feature weights and accuracy  

### **Interleaved Practice**
- Mini-batches are drawn from a fully shuffled dataset  
- Produces stable, smooth learning curves  

Both methods run for **5 epochs** with **batch size = 256**.

---

## 🔍 Feature Tracking

We track two types of features:

### **1. Diagnostic Words**
Single words strongly tied to each topic:

- `government` → World  
- `soccer` → Sports  
- `stock` → Business  
- `space` → Sci/Tech  

### **2. Lexical Sets**
Word groups representing each class:

- **World:** president, minister, war, iraq  
- **Sports:** game, team, coach, season  
- **Business:** market, shares, profit, company  
- **SciTech:** research, software, space, internet  
- **Common words:** news, said, today, new  

We compute **average coefficient values** to study how weights stabilize.

---

# 📊 Key Results

### **1. Accuracy**
- Interleaved training reaches high accuracy quickly  
- Massed training oscillates heavily and stabilizes slowly  

### **2. Diagnostic Word Coefficients**
- Massed practice causes large swings whenever a new class block begins  
- Interleaving leads to smooth, monotonic convergence  

### **3. Lexical Set Coefficients**
- Massed practice shows repeated rises and crashes  
- Interleaving produces stable and consistent values  

**Conclusion:**  
Interleaving leads to more stable, generalizable learning — reflecting cognitive science findings on spaced practice.

---

# ▶️ How to Run the Project

### **1. Clone the repository**
```bash
git clone https://github.com/GAT007/hmlcap.git
cd hmlcap
