import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
#-------------------------------------
#------------------------ THE ADDRESS MAY VARY----------------
#-------------------------------------
df = pd.read_csv(r"C:\Users\control\Desktop\extract_mov\netflix_titles.csv")

# Handle missing values
df['director'] = df['director'].fillna('')
df['cast'] = df['cast'].fillna('')
df['description'] = df['description'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')

# Combine useful text features
df['combined'] = (
    df['description'] + " " + 
    df['listed_in'] + " " + 
    df['cast'] + " " + 
    df['director']
)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping of titles to index (case-insensitive)
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Function for multiple movie recommendations
def recommend_multiple(titles, cosine_sim=cosine_sim):
   
    titles = [t.lower() for t in titles]
    
    indices_list = [indices[t] for t in titles if t in indices]
    if not indices_list:
        return ["No matching titles found."]
    
    sim_scores = sum([cosine_sim[idx] for idx in indices_list]) / len(indices_list)
    
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    movie_indices = [i[0] for i in sim_scores if df['title'].iloc[i[0]].lower() not in titles][:6]
    
    results = df.iloc[movie_indices][['title', 'type', 'release_year', 'listed_in']]
    return results

# User Input
print("\nMovie Recommender Tool:-")
user_input = input("Enter the names of movies/shows you like (separated by commas): ")

# Split input into list and strip spaces
user_titles = [t.strip() for t in user_input.split(",")]

print("\nBased on your watch history entered, we recommend:\n")
print(recommend_multiple(user_titles))
