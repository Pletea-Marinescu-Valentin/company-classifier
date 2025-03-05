# Company Classifier for Insurance Taxonomy

## Overview
This project provides a robust company classifier that maps businesses to relevant insurance taxonomy labels based on their descriptions, business tags, and classifications. The goal is to develop an intelligent system capable of handling various company datasets and accurately associating them with predefined insurance-related categories.

## Approach
To solve this problem, we used a Natural Language Processing (NLP) approach that leverages vectorization and similarity matching techniques. The main steps are:

1. **Data Preprocessing**
   - Convert all text data to lowercase for consistency.
   - Combine multiple fields (description, business tags, sector, category, niche) into a single textual representation for each company.
   - Remove special characters and unnecessary formatting to enhance model accuracy.

2. **Vectorization & Similarity Calculation**
   - Convert company descriptions and taxonomy labels into numerical representations using **TF-IDF Vectorization**.
   - Compute **cosine similarity** between each company and the predefined taxonomy labels.
   - Set a similarity threshold to determine relevant labels for each company.

3. **Label Assignment**
   - Companies are classified into one or more insurance-related labels based on the highest similarity scores.
   - If no similarity surpasses the threshold, the company remains unclassified.

## Key Decisions & Trade-offs
- **TF-IDF over Word Embeddings:**
  - TF-IDF provides a lightweight and effective way to analyze text similarity without requiring a pre-trained language model.
  - It performs well for this specific task where taxonomy labels are short and descriptive.
  
- **Cosine Similarity over Classification Models:**
  - Since this is an unsupervised classification task with no labeled training data, cosine similarity allows us to measure direct textual relevance.
  - Avoids overfitting and unnecessary complexity while still being scalable.

## Running the Code
### Prerequisites
- Python 3.x
- Pandas
- Scikit-learn

### Steps to Execute
1. Install dependencies:
   ```sh
   pip install pandas scikit-learn
   ```
2. Place `ml_insurance_challenge.csv` and `insurance_taxonomy - insurance_taxonomy.csv` in the same directory as the script.
3. Run the script:
   ```sh
   python company_classifier.py
   ```
4. The classified results will be saved in `classified_companies.csv`.

## Results & Learnings
- The model successfully maps most companies to relevant insurance-related labels.
- Certain niche categories might require additional manual curation or an expanded taxonomy list.
- Experimenting with different vectorization techniques could further refine accuracy.

## Conclusion
This solution demonstrates a practical approach to classifying companies within an insurance taxonomy using NLP. The methodology is adaptable and can scale efficiently with larger datasets, making it a strong foundation for further improvements and real-world applications.
