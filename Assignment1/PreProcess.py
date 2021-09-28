import re

import nltk
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import reuters
from typing import List, Set

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


class DateRecognition:

    def __init__(self, pos_tag_list: List[List[str]]):
        self.pos_tag = pos_tag_list
        self.date_list: Set[str] = set()
        self.month = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
                      'August', 'September', 'October', 'November', 'December',
                      'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
        self.date_patterns = "(\\d{4}[-/]?\\d{2}[-/]?\\d{2})" \
                             "|(\\w+\\s\\d{1,2}[a-zA-Z]{2}\\s?,?\\s?\\d{4}?)" \
                             "|(the\\s\\d{1,2}[a-zA-Z]{2}\\sof\\s[a-zA-Z]+)" \
                             "|(the\\s\\w+\\sof\\s\\w+)" \
                             "|(\\w+\\s\\d{1,2}[a-zA-Z]{2})" \
                             "|(\\w+\\s\\d{1,2})"
        self.date_regexp = re.compile(self.date_patterns)

    def data_recognition(self) -> Set[str]:
        # extract different format of dates from the each sentence
        date_recognition_cfg = r"""
            DATE:   {<NNP> <CD> <,>? <CD>}          # E.g. December 12th 2020, December 12th , 2020
                    {<DT> <NN> <IN> <NNP>}          # E.g. the twelfth of December
                    {<DT> <CD> <IN> <NNP>}          # E.g. the 12th of December
                    {<IN> <NNP> <CD>}               # E.g. on December 12th
                    {<NNP> <CD>}                    # E.g. March 30
                    {<IN> <CD>}                     # E.g. in 2020, on 2020/12/12
                    {<IN> <JJ>}                     # E.g. on 2020-12-12
        """
        cp = nltk.RegexpParser(date_recognition_cfg)

        for i in range(len(self.pos_tag)):
            tree = cp.parse(self.pos_tag[i])
            # tree.draw()
            for subtree in tree.subtrees():
                if subtree.label() == 'DATE':
                    tokens = [tup[0] for tup in subtree.leaves()]
                    if '/' in tokens or '-' in tokens:
                        date = ''.join(ch for ch in tokens)
                    else:
                        date = ' '.join(word for word in tokens)

                    # check if date satisfies all conditions
                    validity = self.data_validate(date, tokens)

                    if validity:
                        self.date_list.add(date)

        return self.date_list

    def data_validate(self, date_str: str, tokens: List[str]) -> bool:
        # traverse tokens to check if numbers are valid
        for token in tokens:
            if token not in ['/', '-', ','] and not token.isalnum():
                # meaning that tokens may have float, special characters or non-alphanumeric characters
                return False

        # check if data_str satisfies the date_patterns
        check = self.date_regexp.findall(date_str)
        if len(check) == 0 or check == []:
            return False

        return True


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

            print('\nText Preprocess and proofreading result display as follows:')
            print('\n【Tokenization】')
            print(title_tokens)
            print(body_tokens)

            # 2. sentence splitting
            # ## use built-in tagged sentence (for clarify)
            body_sentences = nltk.sent_tokenize(body)
            print('\n【Sentences Splitting】')
            print(body_sentences)

            # 3. POS tagging
            pos_tags: List[List[str]] = list()
            for body_sentence in body_sentences:
                body_tokens = regexp_tokenizer.tokenize(body_sentence)
                body_pos_tags = nltk.pos_tag(body_tokens)
                pos_tags.append(body_pos_tags)
            print('\n【POS Tagging】')
            print(pos_tags)

            # 4. number normalization

            # 5. date recognition
            dr = DateRecognition(pos_tags)
            dates = dr.data_recognition()  # get a list of detected dates
            print('\n【Date recognition】')
            print(dates)

        except Exception as ex:
            print(ex.args[0])


if __name__ == '__main__':
    PreProcess = TextPreprocess(text)
    tokenizer_types = ['base', 'enhanced']
    PreProcess.text_preprocess(tokenizer_types[1], tokenizer_types)
