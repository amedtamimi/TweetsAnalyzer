import twint
c= twint.Config()
c.Search = 'Amman'
c.Limit =5 
# c.MongoDBurl = "mongodb+srv://tweet:tweet@cluster0.u9gul.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
# c.MongoDBdb = "tweets_db"
# c.MongoDBcollection = "tweets_records"
c.Database = True
twint.run.Search(c)
