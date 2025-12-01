# HML 7651 : Learning and Forgetting in Text Classification  
### By: Manya Jain, Amith Tallanki, Suchitra Yelchuri

Can a high-dimensional linear text classifier reproduce the classic human-learning advantage of interleaved practice and the corresponding catastrophic forgetting observed under massed practice?


Here, we are trying to find out how training schedules affect the learning dynamics of a text classifier.  
We compare 2 training strategies:

- **Massed practice** — model sees one class at a time (blocked training)  
- **Interleaved practice** — model sees mixed classes continuously (shuffled batches)

We are employing the **AG News Classification Dataset**. For this, we analyze the test accuracy and **feature weights change over time**.

Our code is in **`main.ipynb`**.

## Dataset

We use the AG News dataset. You can download it with `kagglehub`.

We have 4 predefined categories that are already available in the dataset: World, Sports, Business, Science/Tech. These are given the labels 0, 1, 2, and 3 respectively. We also included one other category Common which has the label None.

## Preprocessing

We are constructing the text by concatenating every article’s title and description.  
We apply **TF–IDF vectorization**. For this, we use:

- 50,000 max features  
- Unigrams and bigrams  
- English stopword removal  


## Training Methods

Now, we train an `SGDClassifier` (logistic regression using stochastic gradient descent) in 2 schedules:

### **Massed Practice**
- Data is split by class  
- Each epoch iterates: class 0 to 1 to 2 to 3
- Produces large oscillations in feature weights and accuracy  

### **Interleaved Practice**
- Mini-batches are drawn from a fully shuffled dataset  
- Produces stable, smooth learning curves  

Both methods run for **5 epochs** with **batch size 256**.


## Feature Tracking

We try to track 2 types of features:

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

# Key Results

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
Interleaving leads to more stable, generalizable learning. This shows us cognitive science findings on spaced practice.


# How to Run the Project

### **1. Clone the repository**
```bash
git clone https://github.com/GAT007/hmlcap.git
cd hmlcap
