# Natural Language Processing(NLP) Assignments:
Course exercises and coding assignments from "Natural Language Processing with Python" by Lazy Programmer on Udemy.

---

##  Key Libraries
* Data Processing: `NumPy`, `Pandas`
* Classic NLP & Vectorization:** `NLTK`, `Scikit-learn`
* Deep Learning & Neural Networks:** `TensorFlow`, `Keras`
* Linear Algebra & SVD:** `SciPy`

---

##  Repository Structure

Each directory focuses on a core NLP concept.

| Directory | Description | Tools |

* | Article_spinner | An unsupervised text spinner that predicts and replaces words based on the conditional probability of their surrounding left and right context. | Trigram Markov Model, NLTK, NumPy |
* | Cipher_Decryption | An automated decryption engine that uses a Genetic Algorithm (GA) with custom crossover and mutation operators to break substitution ciphers. | Genetic Algorithms, Markov Bigrams, Python |
* | Markov_Model | A dual-purpose Markov engine featuring a first-order (bigram) classifier to identify authors (Frost vs. Poe) and a second-order (trigram) text generator for synthetic poetry. | Markov Chains, Bigrams/Trigrams, NumPy |
* | SVD | A content-based movie recommendation engine that compares raw TF-IDF search against Latent Semantic Analysis (LSA) using Truncated SVD and cosine similarity. | Scikit-learn, TruncatedSVD, Cosine Similarity |
* | Sentiment_Analysis | A comparative sentiment analysis pipeline comparing multi-class (positive/negative/neutral) and binary classifiers optimized with TF-IDF vectorization and Logistic Regression. | Scikit-learn, TF-IDF, Logistic Regression |
* | Spam_detection | A dual-implementation spam filter comparing a hand-coded Multinomial Naive Bayes classifier (using Laplace smoothing and log-likelihoods) against Scikit-learn's MultinomialNb. | NumPy, Scikit-learn, CountVectorizer |
* | Text_Summarization | Extractive text summarizers comparing a normalized TF-IDF sentence weight algorithm against a custom TextRank (PageRank) graph-based model solved via eigenvector analysis. | NumPy, NLTK, Scikit-learn, Graph Theory |
---

##  Key Learning Milestones

* "Vector Space Models & SVD"
  * Bag-of-Words (BoW) & TF-IDF vectorization.
  * Truncated SVD (LSA) for latent semantic dimensionality reduction.

* "Markov Models"
  * Bigram & trigram probability sequence modeling.

* "Graph-Based NLP"
  * TextRank extractive summarization using sentence similarity graphs.
  * Eigenvector decomposition in NumPy to find highly ranked central sentences.

* "Genetic Algorithms"
  * Evolutionary optimization to decrypt monoalphabetic substitution ciphers.
  * Custom chromosome crossover, swap mutations, and log-likelihood fitness.

* "ML & Deep Learning in NLP"
  *
  *

---
