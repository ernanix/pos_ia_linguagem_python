import pandas as pd

# Make display smaller
pd.options.display.max_rows = 10

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(r'./datasets/movielens/users.dat', sep='::',
                      header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(r'./datasets/movielens/ratings.dat', sep='::',
                        header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(r'./datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames)
data = pd.merge(pd.merge(ratings, users), movies)

# 1. Os 10 filmes melhor classificados (com mais de 300 avaliações).
mean_ratings = data.pivot_table('rating', index='title', aggfunc='mean')
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 300]
mean_ratings.loc[active_titles].sort_values(by="rating", ascending=False)[:10]

# 2. Os 10 filmes pior classificados (com mais de 300 avaliações).
mean_ratings.loc[active_titles].sort_values(by="rating")[:10]

# 3. Quantidade de filmes por gênero (considerar apenas o primeiro 
# gênero de cada filme).
movies['first_genre'] = movies['genres'].apply(lambda x: str(x).split('|')[0])
movies.groupby('first_genre').size()

# 4. Os 10 filmes melhor classificados no gênero "comedy".
data = pd.merge(pd.merge(ratings, users), movies)
comedy_movies = data[data['first_genre'] == "Comedy"]
active_titles = comedy_movies['title'].unique()
mean_ratings.loc[active_titles].sort_values(by="rating", ascending=False)[:10]
