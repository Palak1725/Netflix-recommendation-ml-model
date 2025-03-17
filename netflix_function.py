def fav_mov(movie_name):
  list_of_all_titles=df['title'].astype(str).to_list()
  find_close_match= difflib.get_close_matches(movie_name,list_of_all_titles)
  close_match=find_close_match[0]
  index_mov=df[df.title==close_match]["index"].values[0]
  similarity_score=list(enumerate(similarity[index_mov]))
  sorting_similar_mov=sorted(similarity_score,key= lambda x:x[1], reverse=True)
  top10_recomended= sorting_similar_mov[:10]
  print("Top 10 related movies for",movie_name, "are: ")
  k=1
  for i in top10_recomended:
    index=i[0]
    title_of_movie=df[df.index==index]['title'].values
    for i in title_of_movie :
      print(k,title_of_movie)
      k+=1
fav_mov("fast and furious")
