
import sys
import json
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

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

def classify(classifier,X,y,tX,ty):
  print("training started...")

  if classifier == "NB":
    pred = naive_bayesian(X,y,tX)
  elif classifier == "LR":
    pred = logistic_regression(X,y,tX)
  else:
    pred = SVM(X,y,tX)
  
  print("training finished!!!")
  return pred.tolist()

def make_Y(json_docs):
  return [int(doc["sentiment"]) for doc in json_docs]

def make_X(json_docs,voca):
  X = []
  for doc in json_docs:
    words = [w for word in doc["text"] for w,c in word.items()]
    x = [1 if v in words else 0 for v in voca]
    X.append(x)

  return X

def make_voca(bag_of_words,top=10000):
  swords = stopwords.words("english")

  voca = list(sorted(bag_of_words, key=bag_of_words.__getitem__, reverse=True))
  voca = [w for w in voca if w not in swords]
  #threshold = 5
  #voca = [w for w,c in bag_of_words.items() if c >= threshold and w not in swords]
  if len(voca) > top:
    voca = voca[:top]

  return voca

def make_BOW(json_docs):
  bag_of_words = {}

  for doc in json_docs:
    for word in doc["text"]:
      for w,c in word.items():
        if w in bag_of_words.keys():
          bag_of_words[w] += c
        else:
          bag_of_words[w] = c

  return bag_of_words

def write_docs(dic):
  pass

def read_docs(dic):
  json_docs = []

  with open(dic,"r") as f:
    while True:
      line = f.readline()
      if line == "":
        break
      else:
        json_docs.append(json.loads(line))

  return json_docs

def main():
  if len(sys.argv)>=2 and sys.argv[1] == "-h":
    print("usage: python classifier.py {path_to_train_data} {path_to_test_data} {path_to_output_json} {classifer=(NB,LR,SVM)")
    exit(0)

  assert len(sys.argv)>=5, "Please input parameters appropriately. Command python classifier.py -h will help you."
  train_path = sys.argv[1]
  test_path = sys.argv[2]
  output_path = sys.argv[3]
  classifier = sys.argv[4]

  json_docs = read_docs(train_path)
  test_json_docs = read_docs(test_path)
  bag_of_words = make_BOW(json_docs)
  voca = make_voca(bag_of_words)
 
  print("dimension:",len(voca))
  print("classifier:",classifier)

  X = make_X(json_docs,voca)
  y = make_Y(json_docs)
  tX = make_X(test_json_docs,voca)
  ty = make_Y(test_json_docs)
 
  results = classify(classifier,X,y,tX,ty)

  accuracy = sum([1 for i,v in enumerate(results) if v == ty[i]])/len(results)
  print("accuracy:",accuracy)

  for i, doc in enumerate(test_json_docs):
    doc["sentiment"] = results[i]
    del doc["text"]

  with open(output_path,"w") as f:
    for doc in test_json_docs:
      print(json.JSONEncoder().encode(doc),file=f)

main()
