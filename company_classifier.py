import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading CSV Data
company_list = pd.read_csv("ml_insurance_challenge.csv")
insurance_taxonomy = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")

# Preprocess text data
def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""

company_list['combined_text'] = company_list[['description', 'business_tags', 'sector', 'category', 'niche']].astype(str).agg(' '.join, axis=1)
company_list['combined_text'] = company_list['combined_text'].apply(preprocess_text)
insurance_taxonomy['label'] = insurance_taxonomy['label'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer()
company_vectors = vectorizer.fit_transform(company_list['combined_text'])
taxonomy_vectors = vectorizer.transform(insurance_taxonomy['label'])

# Compute similarity scores
similarity_matrix = cosine_similarity(company_vectors, taxonomy_vectors)

# Assign labels based on highest similarity
threshold = 0.1  # Adjust based on experimentation
def assign_labels(similarity_row):
    indices = np.where(similarity_row > threshold)[0]
    return [insurance_taxonomy.iloc[i]['label'] for i in indices]

company_list['insurance_label'] = [assign_labels(row) for row in similarity_matrix]

# Save results
company_list.to_csv("classified_companies.csv", index=False)

print("Classification complete. Results saved to classified_companies.csv")
