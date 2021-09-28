import nltk
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import reuters
from typing import List

print("\nWelcome to the text preprocess and proofreading results program!")
print("\nPlease enter a file name to process")
print("For example:【training/267】")

text = input()


def reformat_body(body: str) -> str:
    """
    Due to the body text may have new lines in the middle of a sentence,
    perform a reformation before processing
    :param body: body content
    :return: one-line reformed body
    """
    reformed_body = ""
    for line in body.split('\n'):
        reformed_body += line.strip() + ' '
    return reformed_body


class TextPreprocess:

    def __init__(self, fileid: str):
        self.fileid = fileid
        self.raw_text = ""
        self.save_file_name = 'reuters-' + self.fileid.replace('/', '-') + '.txt'

    def read_file(self):
        """
        read text file from reuters corpus from NLTK and write it on local
        :return: True if successful, otherwise False
        """
        if self.fileid in reuters.fileids():
            # get the text using the NLTK corpus
            self.raw_text = reuters.raw(self.fileid)
        else:
            raise Exception('ERROR: In reuters corpus【' + str(text) + '】does not exist!')

        # save the raw text on local for backup
        with open(self.save_file_name, "w") as f:
            f.write(self.raw_text)

    def title_content_split(self):
        # replace the html symbols with actual symbols
        self.raw_text = self.raw_text.replace('&lt;', '<')
        self.raw_text = self.raw_text.replace('&gt;', '>')

        # split title and content content
        title, content = self.raw_text.split('\n', 1)

        # check if split is successful
        # check if title is in uppercase
        if title.upper() != title:
            print('WARNING: The text file【' + str(text) + '】does not have a title!')
            # consider all contents as content
            content = title + content
            return "", content
        else:
            return title, content

    def text_preprocess(self, tokenizer_type: str, tokenizer_list: List[str]):
        try:
            # read the text from NLTK
            self.read_file()

            # get the title and body contents
            title, body = self.title_content_split()

            # reformat the body as the text in assignment description
            body = reformat_body(body)

            # 1. tokenization
            if tokenizer_type == tokenizer_list[0]:
                # use basic regular expression tokenizer (not enhanced) from NLTK book chapter 3
                pattern = r'''(?x)              # set flag to allow verbose regexps
                        (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
                      | \w+(?:-\w+)*            # words with optional internal hyphens
                      | \$?\d+(?:\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
                      | \.\.\.                  # ellipsis
                      | [][.,;"'?():-_`]        # these are separate tokens; includes ], [
                      '''
            elif tokenizer_type == tokenizer_list[1]:
                # use enhanced regular expression tokenizer based on basic regular expression tokenizer
                pattern = r'''(?x)                  # set flag to allow verbose regexps
                        (?:[A-Z]\.)+                # abbreviations, e.g. U.S.
                      | \$?\d+(?:,\d+)?(?:\.\d+)?%? # currency or percentages or numbers that include a comma and/or a period e.g. $12.50, 52.25%, 30,000, 3.1415, 1,655.8
                      | \w+(?:-\w+)*                # words with optional internal hyphens e.g. state-of-the-art
                      | \.\.\.                      # ellipsis ...
                      | \'[sS]                      # tokenize "'s" together
                      | [][.,;"'?():-_`]            # these are separate tokens; include ], [
                      | \w+                         # word characters
                      '''
            else:
                raise Exception("ERROR: Tokenizer type【\'" + str(tokenizer_type) + "\'】does not exist in【" + (
                    ', '.join(tokenizer_list)) + '】.')

            regexp_tokenizer = RegexpTokenizer(pattern)
            title_tokens = regexp_tokenizer.tokenize(title)
            body_tokens = regexp_tokenizer.tokenize(body)
            print()

            print('Text Preprocess and proofreading result display as follows:')
            print('\n【Tokenization】')
            print(title_tokens)
            print(body_tokens)

            # 2. sentence splitting
            # ## use built-in tagged sentence (for clarify)
            body_sents = nltk.sent_tokenize(body)
            print('\n【Sentences Splitting】')
            print(body_sents)

            # 3. POS tagging
            pos_tags: List[List[str]] = list()
            for body_sent in body_sents:
                body_tokens = regexp_tokenizer.tokenize(body_sent)
                body_pos_tags = nltk.pos_tag(body_tokens)
                pos_tags.append(body_pos_tags)
            print('\n【POS Tagging】')
            print(pos_tags)
            print()

        except Exception as ex:
            print(ex.args[0])


if __name__ == '__main__':

    PreProcess = TextPreprocess(text)
    tokenizer_types = ['base', 'enhanced']
    PreProcess.text_preprocess(tokenizer_types[1], tokenizer_types)
