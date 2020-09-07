from fugashi import Tagger


def mecab_tokenizer(text):
    tagger = Tagger('-Owakati')
    return [str(word) for word in tagger.parse(text.lower()).split() if str(word).strip()]
