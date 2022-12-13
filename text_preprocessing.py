import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()


def lemmatize_word(word: str) -> str:
    try:
        p = morph.parse(word)[0]
        return p.normal_form
    except:
        return word


def preprocessing(text):
    text = text.lower()
    clear = re.sub(r'[^а-яА-Я]', ' ', text)  # все кроме букв
    clear = re.sub(r"\s+[а-яА-Я]\s+", ' ', clear)  # одиночные буквы
    clear = re.sub(r'\s+', ' ', clear)  # лишние пробелы

    return ' '.join([lemmatize_word(word) for word in clear.split(' ')])