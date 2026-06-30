import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try :
    df = pd.read_csv("raw_skills.csv")
    print("[SUCCESS]Career dataset 'raw_skills.csv' loaded successfully.")
except FileNotFoundError:
    print("[[ERROR] Could not find 'raw_skills.csv'. Make sure it is in your folder!]")

print("---Welcome to DecodeLabs Tech Recommender---")
print("Please input a minimum of three core technical skills or intrests.\n")

skill_1 = input("Enter skill 1: ").strip().lower()
skill_2 = input("Enter skill 2: ").strip().lower()
skill_3 = input("Enter skill 3: ").strip().lower()

user_profile = f"{skill_1} {skill_2} {skill_3}"

print("\n--------------------------------------------------")
print(f"-> Total Career Paths Ingested: {df.shape[0]}")
print(f"-> Raw Ingested User Profile State: ['{user_profile}']")
print("--------------------------------------------------")

print("--- PHASE 2: PROCESS PIPELINE INITIALISED")

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df['skills'])

user_vector = vectorizer.transform([user_profile])

similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

print("[SUCCESS] Vector mapping complete. Text attributes converted to TF-IDF arrays.")
print("[SUCCESS] Similarity scoring engine run complete.\n")

print("RAW CALCULATED SCORES: ")
for idx, score in enumerate(similarity_scores):
    print(f"-> {df['role'].iloc[idx]}: {score:.4f}")
print("--------------------------------------------------")