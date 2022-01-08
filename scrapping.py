import twint
import nest_asyncio
nest_asyncio.apply()

#Utilisationde la bibliotheque Twint pour l'extraction des Tweets 

c = twint.Config()

c.Search = ["Ballon d'or"]       # topic
c.Limit =100000    # number of Tweets to scrape
c.Store_csv = True       # store tweets in a csv file
c.Output = "balondor_tweets.csv"     # path to csv file

twint.run.Search(c)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# =============== 2.8 : REPRESENTATION ===============
    def show(self, n_docs=-1, tri="alphabetique"):
        docs = list(self.id2doc.values())
        if tri == "alphabetique":  # Tri alphabétique
            docs = list(sorted(docs, key=lambda x: x.titre.lower()))[:n_docs]
        elif tri == "numerique":  # Tri temporel
            docs = list(sorted(docs, key=lambda x: x.date))[:n_docs]

        print("\n".join(list(map(repr, docs))))

    def __repr__(self):
        docs = list(self.id2doc.values())
        docs = list(sorted(docs, key=lambda x: x.titre.lower()))

        return "\n".join(list(map(str, docs)))



