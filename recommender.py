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

print("--- PHASE 3: OUTPUT PIPELINE INITIALIZED ---")

# 1. Pair each career index position with its calculated similarity score
# 2. Sort the array in descending order based on the match scores
# 3. Truncate the list to isolate only the top N results (Top 3 paths)
top_n = 3
sorted_indices = similarity_scores.argsort()[::-1][:top_n]

print("[SUCCESS] Descending sorting executed. Matches organized by relevance.")
print("[SUCCESS] Output truncation filter applied. Top-N threshold locked.\n")

print("================ FINAL DIAGNOSTIC REPORT ================")
print("RECOMMENDED CAREER PATHS BASED ON SKILL PREFERENCES:")
print("---------------------------------------------------------")

# Loop through our isolated top indices and format the final output card
for rank, idx in enumerate(sorted_indices, start=1):
    role_name = df['role'].iloc[idx]
    match_percentage = similarity_scores[idx] * 100
    
    print(f"Rank {rank}: {role_name:<20} | Match Score: {match_percentage:.2f}%")

print("=========================================================")
print("[SUCCESS] Production-grade recommendation engine complete!")