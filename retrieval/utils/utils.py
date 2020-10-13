import MeCab
wakati = MeCab.Tagger("-Owakati")

def mecab_tokenizer(text):
    return [str(word) for word in wakati.parse(text).split() if str(word).strip()]

if __name__ == "__main__":
    print(mecab_tokenizer("Please take on me"))