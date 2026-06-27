import pandas as pd

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