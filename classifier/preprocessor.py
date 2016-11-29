
import sys
import json
import nltk

def preprocess_data(input_path, output_path, bi_output_path, limit):
  c = 0

  with open(input_path,"r") as f:
    with open(output_path,"w") as f2:
      with open(bi_output_path,"w") as f3:
        while c < limit:
          c+=1
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
          output = dict([("serial", json_line["serial"]), ("sentiment", json_line["sentiment"]), ("text", words)])
          print(json.JSONEncoder().encode(output),file=f2)

          bigrams = nltk.bigrams(tokens)
          bi_words = [dict([(' '.join(k), v)]) for k,v in nltk.FreqDist(bigrams).items()]

          output = dict([("serial", json_line["serial"]), ("sentiment", json_line["sentiment"]), ("text", bi_words)])
          print(json.JSONEncoder().encode(output),file=f3)

def main():
  assert len(sys.argv) >=5, "Please enter input path, output path, bi_output_path, limit"
  input_path = sys.argv[1] 
  output_path = sys.argv[2]
  bi_output_path = sys.argv[3]
  limit = int(sys.argv[4])

  preprocess_data(input_path, output_path, bi_output_path, limit)

main()
