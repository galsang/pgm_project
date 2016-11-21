
import json
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

swords = stopwords.words("english")
bag_of_words = {}
top = 10000
threshold = 5
json_docs = []

def preprocess_data(limit=100):
  c = 0

  with open("dataset1.json","r") as f:
    with open("data.json","w") as f2:
      while c < limit:
        c += 1
        line = f.readline()
        if line == "":
          break
        else:
          json_line = json.loads(line)
    
        tokens = nltk.word_tokenize(json_line["text"])
        tokens = [token.lower() for token in tokens]
        unique_tokens = list(set(list(tokens)))
        words = [dict([(token, tokens.count(token))]) for token in unique_tokens]
        json_line["sentiment"] = int((json_line["sentiment"] + 1) > 0)
        output = dict([("serial", json_line["serial"]), ("sentiment", json_line["sentiment"]), ("words", words)])
        print(json.JSONEncoder().encode(output),file=f2)

def SVM(X,y,tX):
  clf = svm.SVC(kernel="linear")
  clf.fit(X,y)
  
  return clf.predict(tX)

def naive_bayesian(X,y,tX):
  clf = MultinomialNB()
  clf.fit(X,y)
  
  return clf.predict(tX)

def logistic_regression(X,y,tX):
  clf = LogisticRegression()
  clf.fit(X,y)

  return clf.predict(tX)

def main():
  with open("data.json","r") as f:
    while True:
      line = f.readline()
      if line == "":
        break
      else:
        json_docs.append(json.loads(line))

  for doc in json_docs:
    for word in doc["words"]:
      for w,c in word.items():
        if w in bag_of_words.keys():
          bag_of_words[w] += c
        else:
          bag_of_words[w] = c

  #voca = [w for w,c in bag_of_words.items() if c >= threshold and w not in swords]
  voca = list(sorted(bag_of_words, key=bag_of_words.__getitem__, reverse=True))
  voca = [w for w in voca if w not in swords]
  if len(voca) > top:
    voca = voca[:top]

  print("dimension:",len(voca))

  X = []
  for doc in json_docs:
    words = [w for word in doc["words"] for w,c in word.items()]
    x = [1 if v in words else 0 for v in voca]
    X.append(x)
      
  y =[int(doc["sentiment"]) for doc in json_docs]
  
  boundary = int(limit * 0.8)
  pred_SVM = SVM(X[:boundary],y[:boundary],X[boundary:limit]).tolist()
  pred_NB = naive_bayesian(X[:boundary],y[:boundary],X[boundary:limit]).tolist()
  pred_LR = logistic_regression(X[:boundary],y[:boundary],X[boundary:limit]).tolist()

  answer = y[boundary:limit]

  accuracy_SVM = sum([1 for i,v in enumerate(pred_SVM) if v == answer[i]])/len(pred_SVM)
  accuracy_NB = sum([1 for i,v in enumerate(pred_NB) if v == answer[i]])/len(pred_NB)
  accuracy_LR = sum([1 for i,v in enumerate(pred_SVM) if v == answer[i]])/len(pred_LR)

  #print("prediction:",prediction)
  #print("answer:",answer)
  print("SVM accuracy:",accuracy_SVM)
  print("naive bayes accuracy:",accuracy_NB)
  print("logistic regression accuracy:",accuracy_LR)

limit = 10000
preprocess_data(limit)
print("data preprocessing finished")
main()
