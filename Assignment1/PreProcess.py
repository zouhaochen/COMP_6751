"""
Project 1 for COMP 6751 Natural Language Analyst
Text Pre-processing Pipeline and Proofreading Results Program
Be sure to read the project description page for further information about the expected behavior of the program
@author: Haochen Zou
"""

import re

import nltk
from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
from typing import List, Set


print("\nWelcome to the text preprocess and proofreading results program!")
print("\nPlease enter a file name to process")
print("For example: training/267")

text = input()


def body_reform(body: str) -> str:
    """
    Inputs:
      body  - body content of text file
    Output:
      A body after reformation without lines between sentences
    """
    reformed_body = ""
    for line in body.split('\n'):
        reformed_body += line.strip() + ' '
    return reformed_body


class DateRecognition:

    def __init__(self, pos_tag_list: List[List[str]]):
        self.pos_tag = pos_tag_list
        self.date_list: Set[str] = set()

        # date information month definition
        self.month = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
                      'August', 'September', 'October', 'November', 'December',
                      'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

        # date information format definition
        self.date_format = "(\\d{4}[-/]?\\d{2}[-/]?\\d{2})" \
                             "|(\\w+\\s\\d{1,2}[a-zA-Z]{2}\\s?,?\\s?\\d{4}?)" \
                             "|(the\\s\\d{1,2}[a-zA-Z]{2}\\sof\\s[a-zA-Z]+)" \
                             "|(the\\s\\w+\\sof\\s\\w+)" \
                             "|(\\w+\\s\\d{1,2}[a-zA-Z]{2})" \
                             "|(\\w+\\s\\d{1,2})"
        self.date_regexp = re.compile(self.date_format)

    def date_recognition(self) -> Set[str]:
        # date recognition context free grammar definition
        date_recognition_cfg = r"""
            DATE:   {<NNP> <CD> <,>? <CD>}          # October 1st 2021
                    {<DT> <CD> <IN> <NNP>}          # the 1st of October
                    {<DT> <NN> <IN> <NNP>}          # the first of October
                    {<IN> <NNP> <CD>}               # on October 1st
                    {<NNP> <CD>}                    # October 1st
                    {<IN> <CD>}                     # in 2021
                    {<IN> <JJ>}                     # on 2021-10-1
        """
        date_information = nltk.RegexpParser(date_recognition_cfg)

        # draw a parse tree for date information
        for i in range(len(self.pos_tag)):
            tree = date_information.parse(self.pos_tag[i])
            for subtree in tree.subtrees():
                if subtree.label() == 'DATE':
                    tokens = [tup[0] for tup in subtree.leaves()]
                    if '/' in tokens or '-' in tokens:
                        date = ''.join(ch for ch in tokens)
                    else:
                        date = ' '.join(word for word in tokens)

                    # validate date information check
                    validate = self.date_validate(date, tokens)
                    if validate:
                        self.date_list.add(date)

        return self.date_list

    def date_validate(self, date_string: str, tokens: List[str]) -> bool:
        # validate date information check method
        for token in tokens:
            if token not in ['/', '-', ','] and not token.isalnum():
                return False

        # validate date string in date format
        check = self.date_regexp.findall(date_string)
        if len(check) == 0 or check == []:
            return False

        return True


def date_parse(text_date: Set[str]):

    # date parse context free grammar definition
    date_parse_cfg = nltk.CFG.fromstring("""
        DATE -> IN YEAR SEP MONTH_NUM SEP DAY | YEAR SEP MONTH_NUM SEP DAY | MONTH_STR DAY SEP YEAR | MONTH_STR DAY | MONTH_STR YEAR | IN MONTH_STR NN_NUM | MONTH_STR NN_NUM YEAR | MONTH_STR NN_NUM SEP YEAR | MONTH_STR NN_NUM | DT NN_STR IN MONTH_STR | DT NN_NUM IN MONTH_STR | IN YEAR | IN MONTH_STR YEAR
        DIGIT -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        YEAR -> DIGIT DIGIT DIGIT DIGIT
        MONTH_NUM -> DIGIT | DIGIT DIGIT
        MONTH_STR -> "January" | "February" | "March" | "April" | "May" | "June" | "July" | "August" | "September" | "October" | "November" | "December"
        DAY -> DIGIT | DIGIT DIGIT
        NN_NUM -> "1st" | "2nd" | "3rd" | "4th" | "5th" | "6th" | "7th" | "8th" | "9th" | "10th" | "11th" | "12th" | "13th" | "14th" | "15th" | "16th" | "17th" | "18th" | "19th" | "20th" | "21st" | "22nd" | "23rd" | "24th" | "25th" | "26th" | "27th" | "28th" | "29th" | "30th" | "31st"
        NN_STR -> "first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth" | "tenth" | "eleventh" | "twelfth" | "thirteenth" | "fourteenth" | "fifteenth" | "sixteenth" | "seventeenth" | "eighteenth" | "nineteenth" | "twentieth" | "twenty-first" | "twenty-second" | "twenty-third" | "twenty-fourth" | "twenty-fifth" | "twenty-sixth" | "twenty-seventh" | "twenty-eighth" | "twenty-ninth" | "thirtieth" | "thirty-first"
        DT -> "the"
        IN -> "of" | "in" | "on"
        SEP -> "/" | "-" | ","
        """)
    date_parser = nltk.ChartParser(date_parse_cfg)

    # search token according to the date information format
    for date in text_date:
        if date.find('/') != -1 or date.find('-') != -1:
            tokens = [ch for ch in date]
        else:
            tokens = []
            for t in date.split():
                if t.isnumeric():
                    tokens.extend([num for num in t])
                else:
                    tokens.append(t)

        for tree in date_parser.parse(tokens):
            print(tree)
            tree.draw()


class DateParser:

    # initialize date parser
    def __init__(self, sentences: List[str], pos_tag_list: List[List[str]]):
        self.date_sentence = sentences
        self.pos_tag = pos_tag_list


class TextPreprocess:

    def __init__(self, fileid: str):
        self.fileid = fileid
        self.raw_text = ""
        self.save_file_name = 'reuters:' + self.fileid.replace('/', '-') + '.txt'

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
            body = body_reform(body)

            # tokenization
            if tokenizer_type == tokenizer_list[0]:
                # use based regular expression tokenizer (not enhanced) from NLTK book chapter 3
                pattern = r'''(?x)              # set flag to allow verbose regexps
                        (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
                      | \w+(?:-\w+)*            # words with optional internal hyphens
                      | \$?\d+(?:\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
                      | \.\.\.                  # ellipsis
                      | [][.,;"'?():-_`]        # these are separate tokens; includes ], [
                      '''
            elif tokenizer_type == tokenizer_list[1]:
                # use improved regular expression tokenizer based on basic regular expression tokenizer
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

            print('\nText Preprocess and proofreading results display as follows:')
            print('\n【Tokenization】')
            print(title_tokens)
            print(body_tokens)

            # sentence splitting
            # ## use built-in tagged sentence (for clarify)
            body_sentences = nltk.sent_tokenize(body)
            print('\n【Sentences Splitting】')
            print(body_sentences)

            # POS tagging
            pos_tags: List[List[str]] = list()
            for body_sentence in body_sentences:
                body_tokens = regexp_tokenizer.tokenize(body_sentence)
                body_pos_tags = nltk.pos_tag(body_tokens)
                pos_tags.append(body_pos_tags)
            print('\n【POS Tagging】')
            print(pos_tags)

            # date recognition
            date_recognition = DateRecognition(pos_tags)
            date = date_recognition.date_recognition()
            print('\n【Date Recognition】')
            print(date)

            # date parsing
            print('\n【Date Parsing】')
            date_parse(date)

        except Exception as ex:
            print(ex.args[0])


if __name__ == '__main__':
    PreProcess = TextPreprocess(text)
    tokenizer_types = ['basic', 'improved']
    PreProcess.text_preprocess(tokenizer_types[1], tokenizer_types)
