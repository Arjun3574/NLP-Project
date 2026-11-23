# NLP-Project
# Multilingual Customer Complaint Routing & Hybrid LLM Response System

**From multilingual social complaints to department routing, triage, and dual LLM responses**  
**Department:** CSE-AIML | **Institution:** MIT Academy of Engineering (MIT AOE)

### Team Members
* **Himanshu Gharde** — PRN: `202401110007`
* **Arjun Vitthal Mankuskar** — PRN: `202401110038`

**Tech Stack:** `Python 3.13` | `Google Colab` | `Scikit-learn` | `spaCy` | `NLTK` | `Qwen2.5` | `Gemini` | `Gradio`

---

## 1. Project Overview / Abstract
Customer support teams face high volumes of multilingual complaints requiring rapid sorting and empathetic responses. This system combines **Bigram TF-IDF + Logistic Regression** routing, **VADER** urgency scoring, **spaCy Transformer NER**, **LDA** topic modeling, and a hybrid **Local Qwen2.5 + Cloud Gemini** LLM pipeline inside an interactive **Gradio** web interface.

---

## 2. Problem Statement
Manual ticket routing across departments is slow and error-prone. Traditional keyword rules fail to capture noisy social media text, regional languages, and emotional urgency, while relying solely on cloud LLMs increases API costs and exposes private internal triage notes.

---

## 3. Objectives
1. Classify customer complaints across **5 departments** with **>85% accuracy**.
2. Score sentiment polarity to assign ticket urgency and priority levels.
3. Extract organizations, locations, dates, and monetary values via **Transformer NER**.
4. Pair **local Qwen2.5** internal agent briefs with **cloud Gemini** customer replies.

---

## 4. Literature Review
* **TF-IDF + Naive Bayes:** Fast statistical baseline for text classification, but limited in capturing multi-word phrase context and handling class imbalance.
* **VADER:** Rule-based lexicon model tailored for social-media sentiment intensity analysis without extra training overhead.
* **spaCy NER (`en_core_web_trf`):** Transformer-based pipeline for accurate contextual entity extraction.
* **Research Gap Addressed:** Bridges classical ML department routing with dual-role local and cloud LLM response generation.

---

## 5. Dataset
* **Source:** Kaggle Customer Support on Twitter Dataset (`twcs.csv` — **2,811,774** raw tweets).
* **Filtering & Sampling:** Filtered to **787,346** initial inbound customer complaints (`inbound == True` and `in_response_to_tweet_id` is null) across **30 brand support handles**, sampled to **150,000** records, and cleaned (`>= 5` words) to **145,065** samples (**116,052** train / **29,013** stratified test):

| Department | Sampled Rows | Test Support | Mapped Brand Handles |
| :--- | :---: | :---: | :--- |
| **Tech Support** | 51,717 | 10,023 | `@AppleSupport`, `@XboxSupport`, `@AskPlayStation`, `@SpotifyCares`, `@Hulu_Support`, `@MicrosoftHelps` |
| **Travel** | 45,729 | 8,915 | `@Delta`, `@AmericanAir`, `@British_Airways`, `@SouthwestAir`, `@Uber_Support`, `@VirginAtlantic` |
| **Retail** | 24,496 | 4,770 | `@AmazonHelp`, `@AskTarget`, `@Tesco`, `@Sainsburys`, `@AsdaServiceTeam`, `@iHelpBestBuy` |
| **Telecom** | 19,605 | 3,717 | `@SprintCare`, `@VerizonSupport`, `@TMobileHelp`, `@ComcastCares`, `@Ask_Spectrum`, `@VirginMedia` |
| **Billing** | 8,453 | 1,588 | `@ChaseSupport`, `@Ask_WellsFargo`, `@BofA_Help`, `@AskPayPal`, `@AskAmex`, `@AskCiti` |

---

## 6. NLP Preprocessing
1. **Language Detection:** Automatic script/language identification via `langdetect`.
2. **English Translation:** Translates non-English complaints (`Hindi`, `Marathi`) to English via `GoogleTranslator` (`deep-translator`) with a `Gemini` API fallback and empty-output guard.
3. **Lowercase Normalization:** Standardizes all text tokens to lowercase.
4. **Regex Cleaning:** Removes URLs (`http\S+|www\S+`), `@mentions` (preventing brand-handle data leakage), agent tags (`\^\w+`), punctuation/special characters, and extra whitespace.
5. **Short Tweet Filtering:** Drops noisy or incomplete tweets with fewer than **5 words**.
6. **Bigram TF-IDF Vectorization:** Extracts unigrams and bigrams (`ngram_range=(1, 2)`, `max_features=25000`, `min_df=3`, `sublinear_tf=True`) with English stop-word removal.

---

## 7. Methodology

                                            +--> Classification (TF-IDF + Logistic Regression) --> Local Qwen2.5 (Internal Brief) --+
    Complaint Input --> Language Detection --> Regex Cleaning +                                                                                               +--> Gradio Web Interface
      (EN / HI / MR)      & Translation                       +--> Triage & Extraction (VADER + spaCy NER) --------------> Cloud Gemini (Customer Reply) --+

    [Offline Dataset Analysis]: Latent Dirichlet Allocation (LDA) Topic Modeling (5 Topics)

---

## 8. System Architecture

| Layer | Components & Data Flow |
| :--- | :--- |
| **Input Layer** | Gradio Complaint Input (`English`, `Hindi`, `Marathi`) |
| **Processing Layer** | Translation (`langdetect` + `GoogleTranslator`) -> `TF-IDF + Logistic Regression` -> `VADER` Sentiment -> `spaCy NER` |
| **Generation Layer** | **Local Qwen2.5** (Private Internal Agent Brief) + **Cloud Gemini** (Empathetic Customer Reply) |
| **Offline Analysis** | **LDA Topic Modeling** (5 topics trained on 50,000 customer tweets) |

---

## 9. Implementation & Experimental Setup

### Implementation
* **Language & Environment:** Python 3.13, Google Colab (NVIDIA T4 GPU), Gradio Blocks
* **Libraries:** `scikit-learn`, `nltk`, `spaCy`, `langdetect`, `deep-translator`, `transformers`, `torch`, `google-genai`, `joblib`
* **Models:**
  * **Department Classifier:** Bigram `TfidfVectorizer` + `LogisticRegression(max_iter=1000, C=2.0)`
  * **Baseline Classifier:** `TfidfVectorizer(max_features=5000)` + `MultinomialNB()`
  * **Sentiment & Priority:** NLTK `SentimentIntensityAnalyzer` (VADER compound score thresholds)
  * **NER & Topic Modeling:** `en_core_web_trf` and `LatentDirichletAllocation(n_components=5)`
  * **Hybrid LLMs:** `Qwen/Qwen2.5-0.5B-Instruct` (Local) + `gemini-3.5-flash-lite` (Cloud API)

### Experimental Setup
* **Train / Test Split:** 80% Training (`116,052`) / 20% Stratified Testing (`29,013` complaints)
* **LDA Corpus:** 50,000 sampled customer complaints
* **Evaluation Metrics:** Accuracy, Precision, Recall, Macro F1-Score, Weighted F1-Score, and Multilingual End-to-End Validation

---

## 10. Results

| Overall Accuracy | Weighted F1-Score | Macro Avg F1-Score | Evaluated Test Samples |
| :---: | :---: | :---: | :---: |
| **86.63%** | **0.86** | **0.82** | **29,013** |

### Department-Wise Classification Report (Logistic Regression)

| Department | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Travel** | 0.89 | 0.91 | **0.90** | 8,915 |
| **Tech Support** | 0.87 | 0.92 | **0.89** | 10,023 |
| **Retail** | 0.85 | 0.84 | **0.85** | 4,770 |
| **Telecom** | 0.86 | 0.77 | **0.82** | 3,717 |
| **Billing** | 0.74 | 0.58 | **0.65** | 1,588 |
| **Weighted Avg** | **0.86** | **0.87** | **0.86** | **29,013** |

### Multilingual & Topic Discovery Highlights
* **Multilingual Testing:** Marathi complaint (`"माझी डेल्टा एअरलाइन्सची फ्लाइट रद्द झाली आणि माझी बॅग हरवली आहे."`) translated into English and classified as **Travel** with **99.99%** model confidence, alongside English (`en`) and Hindi (`hi`) end-to-end tests.
* **Discovered LDA Topics (50,000 Tweets):**
  1. *Connectivity & Orders:* `internet, going, order, just, want, guys, days, like`
  2. *Accounts & Cards:* `account, help, dont, email, card, hi, does, working`
  3. *Flights & Customer Service:* `flight, service, trying, im, customer, need, help, time`
  4. *Mobile & OS Updates:* `phone, app, iphone, help, update, fix, ios, new`
  5. *Store & Hardware Issues:* `having, store, im, issues, battery, received, 10, minutes`

---

## 11. Analysis
* **Bigram TF-IDF + Logistic Regression** achieved **86.63%** accuracy compared with **82.84%** for **Multinomial Naive Bayes**.
* Logistic Regression improved classification accuracy by **3.79 percentage points** over the Naive Bayes baseline.
* **Billing** had the lowest F1-score (**0.65**), indicating scope for improvement in minority-class classification.
* The **Hybrid LLM architecture** keeps internal agent triage briefs private and zero-cost locally via Qwen2.5 while reserving cloud Gemini calls for customer-facing responses.

---

## 12. Key Highlights
* **86.63%** multi-class routing accuracy (+3.79 percentage points over Naive Bayes).
* **Hybrid Local + Cloud LLM** workflow separating internal agent notes from external customer replies.
* End-to-end **English, Hindi, and Marathi** support with symbol-only input validation.
* Automated sentiment urgency triage (`High` / `Medium` / `Normal`), Transformer NER, and offline LDA topic discovery.

---

## 13. Applications
* E-commerce & Retail Support Helpdesks
* Banking & FinTech Complaint Triage
* Airlines & Telecom Call Centers
* Enterprise CRM Ticket Automation

---

## 14. Conclusion
The system effectively automates multilingual complaint classification, urgency triage, entity extraction, and dual-audience response generation. Combining Bigram TF-IDF + Logistic Regression (`86.63%` accuracy) with VADER, spaCy NER, local Qwen2.5, and cloud Gemini delivers a fast, privacy-aware support pipeline.

---

## 15. Limitations
* Handle-based labels create minor cross-department topic overlap (e.g., billing disputes tweeted to retail or telecom handles).
* The Billing class has smaller dataset support compared to Tech Support and Travel.
* Local Qwen2.5 inference requires a GPU for low-latency generation.

---

## 16. Future Work
* Fine-tune a Transformer classifier (`DistilBERT`) to improve contextual routing on ambiguous complaints.
* Balance and expand minority Billing training samples.
* Integrate directly with live CRM ticketing APIs (Zendesk, Freshdesk, Jira).

---

## How to Run

### 1. Google Colab (Recommended)
1. Upload `NLP_Project.ipynb` to Google Colab and enable **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Add your Google AI Studio key in Colab **Secrets** (left sidebar key icon) as `GEMINI_API_KEY`.
3. Place `twcs.csv` in `/content/drive/MyDrive/twcs.csv` and run all cells top-to-bottom to train the model and launch the public `.gradio.live` demo link.

### 2. Local Execution
    pip install spacy gradio pandas scikit-learn nltk deep-translator langdetect google-genai transformers accelerate joblib
    python -m spacy download en_core_web_sm
    export GEMINI_API_KEY="your_api_key_here"
    python app.py

---

## Repository Structure
* `NLP_Project.ipynb` — Complete training, evaluation, LDA, hybrid LLM, and Gradio notebook
* `app.py` — Standalone Gradio inference application using saved `nlp_model.pkl`
* `requirements.txt` — Python dependencies
* `NLP Poster.png` — Academic project poster

---

## References
* Kaggle Customer Support on Twitter Dataset (`twcs.csv`)
* Pedregosa et al. (2011) — *Scikit-learn: Machine Learning in Python*
* Hutto & Gilbert (2014) — *VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text*
* Blei et al. (2003) — *Latent Dirichlet Allocation (LDA)*
* Qwen2.5 (`Qwen/Qwen2.5-0.5B-Instruct`) & Google Gemini API (`gemini-3.5-flash-lite`)
