import pandas as pd
# Make display smaller
pd.options.display.max_rows = 10

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('datasets/movielens/users.dat', sep='::',
                      header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('datasets/movielens/ratings.dat', sep='::',
                        header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames)

data = pd.merge(pd.merge(ratings, users), movies)

mean_ratings = data.pivot_table('rating', index='title',
                                 columns='gender', aggfunc='mean')

ratings_by_title = data.groupby('title').size()

ratings_by_title[:10]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles

# Select rows on the index
mean_ratings = mean_ratings.loc[active_titles]

top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
top_female_ratings[:10]

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]

sorted_by_diff[::-1][:10]

# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
# Filter down to active_titles
rating_std_by_title = rating_std_by_title.loc[active_titles]
# Order Series by value in descending order
rating_std_by_title.sort_values(ascending=False)[:10]


###########
"""
Trabalhe com o Pandas sobre a base de dados MovieLens e retire novas informações. 
Por exemplo: 
1. Os 10 filmes melhor classificados (com mais de 300 avaliações); 
2. Os 10 filmes pior classificados (com mais de 300 avaliações);
3. Quantidade de filmes por gênero (considerar apenas o primeiro gênero de cada filme); 
4. Os 10 filmes melhor classificados no gênero “comedy”.

"""
data['main_genre'] = data['genres'].str.split('|',expand=True)[0]

mean_ratings300 = data.pivot_table('rating', index='title', aggfunc='mean')

active_titles300 = ratings_by_title.index[ratings_by_title >= 300]

mean_ratings300 = mean_ratings300.loc[active_titles300]

##1
top_best_ratings300 = (mean_ratings300.sort_values(by='rating',ascending =False))[:10]
##2
top_worst_ratings300 = (mean_ratings300.sort_values(by='rating',ascending =True))[:10]
##3
by_genre = data.groupby('main_genre').size()
##4
comedy_movies = data[data['main_genre'].str.contains('Comedy')]['title'].drop_duplicates().array
top_comedy = (((data.pivot_table('rating',index='title',aggfunc='mean'))
               .loc[comedy_movies])
               .sort_values(by='rating',ascending=False))[:10]


