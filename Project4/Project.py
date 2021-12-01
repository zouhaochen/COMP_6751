"""
Project 4 for COMP 6751 Natural Language Analyst
Sentence Analysis Development Program
Be sure to read the project description page for further information about the expected behavior of the program
@author: Haochen Zou (Luke) 40158179
"""

import re
import math
import nltk
from nltk import word_tokenize
from nltk import load_parser, FeatureEarleyChartParser
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from collections import Counter


class SentenceParser:

    def __init__(self, grammar_file: str, print_parse_tree: bool = False,
                 draw_parse_tree: bool = False, save_parse_tree: bool = False):
        """
        sentence tree parser initializer
        Input:
          self - text file sentence content
          grammar_file - sentiment analysis grammar file
          print_parse_tree - print parse tree result
          draw_parse_tree - draw parse tree result
          save_parse_tree - save parse tree result
        Output:
          parser_initializer - initialize parse tree
          print_result - print parse tree
          draw_result - draw parse tree
          save_result - save parse tree
          tree_number - parse tree serial number
        """
        self.parser_initializer = load_parser(grammar_file, trace=0, parser=FeatureEarleyChartParser)
        self.print_result = print_parse_tree
        self.draw_result = draw_parse_tree
        self.save_result = save_parse_tree
        self.tree_number = 1

    def parse(self, token: List[str]) -> Tuple[list, Dict[str, List[str]]]:
        """
        sentence tree parser initializer
        Input:
          self - text file sentence content
          token - sentence pos token
        Output:
          sentiment_value - sentence sentiment value
          parse_tree - sentence parse tree
        """
        sentiment_value = []
        parse_tree: Dict[str, List[str]] = defaultdict(list)

        # parse sentence in parsing tree with token
        for tree in self.parser_initializer.parse(token):
            if self.print_result:
                print(tree)
            if self.draw_result:
                tree.draw_result()

            # apply sentiment value to feature
            sentiment_label = tree.label()['SENTIMENT']

            # sentiment values are defined as negative, positive, and neutral
            if sentiment_label in ['negative', 'positive', 'neutral']:
                sentiment_value.append(sentiment_label)
                parse_tree[sentiment_label].append(str(tree))
            self.tree_number += 1

        # append null if no sentiment value feature detected
        if len(sentiment_value) == 0:
            sentiment_value.append('null')
            parse_tree['null'].append('(null)')

        return sentiment_value, parse_tree


class SentenceReader:

    def __init__(self):
        """
        sentence content reader
        Input:
          self - text file sentence content
        """
        self.sentence_positive = []
        self.sentence_negative = []
        self.sentence_neutral = []

        # sentence file path
        sentence_positive_file = 'positive.txt'
        sentence_negative_file = 'negative.txt'
        sentence_neutral_file = 'neutral.txt'

        with open(sentence_positive_file, "r") as reader:
            self.sentence_positive = reader.readlines()
        self.sentence_positive = [sent.rstrip() for sent in self.sentence_positive]

        with open(sentence_negative_file, "r") as reader:
            self.sentence_negative = reader.readlines()
        self.sentence_negative = [sent.rstrip() for sent in self.sentence_negative]

        with open(sentence_neutral_file, "r") as reader:
            self.sentence_neutral = reader.readlines()
        self.sentence_neutral = [sent.rstrip() for sent in self.sentence_neutral]

    def read_negative_sentence(self) -> List[str]:
        """
        negative sentence content reader
        Input:
          self - text file sentence content
        Output:
          sentence_negative - sentence with negative sentiment value
        """
        return self.sentence_negative

    def read_positive_sentence(self) -> List[str]:
        """
        positive sentence content reader
        Input:
          self - text file sentence content
        Output:
          sentence_positive - sentence with positive sentiment value
        """
        return self.sentence_positive

    def read_neutral_sentence(self) -> List[str]:
        """
        neutral sentence content reader
        Input:
          self - text file sentence content
        Output:
          sentence_neutral - sentence with neutral sentiment value
        """
        return self.sentence_neutral


class AfinnAndSsap:

    def __init__(self):
        """
        affin and ssap performance initializer
        Input:
          self - text file sentence content
        """
        # affin file path
        afinn_file_name = 'AFINN-111.txt'
        self.afinn = dict(map(lambda ws: (ws[0], int(ws[1])), [ws.strip().split('\t') for ws in open(afinn_file_name)]))
        self.pattern_split = re.compile(r"\W+")

        # ssap performance initialization
        self.polarity = []
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.true_neutral = 0
        self.false_neutral = 0

    def sentiment(self, text):
        """
        attain sentiment value of sentence
        Input:
          self - text file sentence content
        Output
          word_sentiment_value - sentence word sentiment value
        """
        sentence_word = self.pattern_split.split(text.lower())
        word_sentiment = list(map(lambda word: self.afinn.get(word, 0), sentence_word))

        # individual word sentiment value weight
        if word_sentiment:
            numerator = sum(word_sentiment)
            sentiment_sum = len(word_sentiment)
            if sentiment_sum == 0:
                word_sentiment_value = 0
            else:
                denominator = math.sqrt(sentiment_sum)
                word_sentiment_value = float(numerator) / denominator
        else:
            word_sentiment_value = 0
        return word_sentiment_value

    def predict_value(self, text: str, original_value: str):
        """
        predict sentiment value of sentence
        Input:
          self - text file sentence content
          ground_value - ground sentiment value of sentence
        """
        sentiment_score = self.sentiment(text)

        if original_value == 'positive':
            if sentiment_score > 0:
                self.true_positive += 1
            elif sentiment_score == 0:
                self.false_neutral += 1
            elif sentiment_score < 0:
                self.false_negative += 1

        elif original_value == 'neutral':
            if sentiment_score > 0:
                self.false_positive += 1
            elif sentiment_score == 0:
                self.true_neutral += 1
            elif sentiment_score < 0:
                self.false_negative += 1
        elif original_value == 'negative':
            if sentiment_score > 0:
                self.false_positive += 1
            elif sentiment_score == 0:
                self.false_neutral += 1
            elif sentiment_score < 0:
                self.true_negative += 1

        self.polarity.append(sentiment_score)

    def performance_result(self):
        """
        sentiment performance result view
        Input:
          self - text file sentence content
        """
        print('True Positive\t', self.true_positive, "\tsentences")
        print('True Negative\t', self.true_negative, "\tsentences")
        print('True Neutral\t', self.true_neutral, "\tsentences")
        print('False Positive\t', self.false_positive, "\tsentences")
        print('False Negative\t', self.false_negative, "\tsentences")
        print('False Neutral\t', self.false_neutral, "\tsentences")


def part_of_speech_tagging(words: List[str]) -> List[Tuple[str, str]]:
    """
    sentiment pos tagger
    Input:
      word - sentence content word
    Output:
      sentence_pos_tag - sentence word pos result
    """
    sentence_pos_tag = nltk.pos_tag(words[:-1])
    return sentence_pos_tag


class SentimentPipeline:

    def __init__(self, earley_parser: SentenceParser, sentence_lexicon: SentenceReader):
        """
        sentiment value initializer
        Input:
          self - text file sentence content
          earley_parser - earley parser initializer
          sentence_lexicon - load sentence content
        """
        self.sentiment_parser = earley_parser
        self.sentiment_lexicon = sentence_lexicon
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.true_neutral = 0
        self.false_neutral = 0
        self.original = AfinnAndSsap()

    def sentence_sentiment_parse(self, token_list: List[str]) -> Tuple[List[Tuple[Any, int]], Dict[str, str]]:
        """
        sentence sentiment value parser
        Input:
          self - text file sentence content
          token_list - attach pos tag token list
        Output:
          parse_tree - sentence sentiment parse tree
        """
        sentiment_label, parse_tree = self.sentiment_parser.parse(token_list[:-1])
        return Counter(sentiment_label).most_common(), parse_tree

    def sentiment_analysis(self) -> None:
        """
        sentiment value pipeline run
        Input:
          self - text file sentence content
        """
        try:

            print("Sentence Sentiment Analysis Program\n")
            print("Sentence Sentiment Analysis Start!")

            # load and analysis positive sentence
            for sentence_positive_sentiment in self.sentiment_lexicon.read_positive_sentence():
                sentence_word = word_tokenize(sentence_positive_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                self.output_results(sentence_positive_sentiment, 'positive', sentence_sentiment_value, sentence_parse_tree)
                self.original.predict_value(sentence_positive_sentiment, 'positive')

            # load and analysis negative sentence
            for sentence_negative_sentiment in self.sentiment_lexicon.read_negative_sentence():
                sentence_word = word_tokenize(sentence_negative_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                self.output_results(sentence_negative_sentiment, 'negative', sentence_sentiment_value, sentence_parse_tree)
                self.original.predict_value(sentence_negative_sentiment, 'negative')

            # load and analysis neutral sentence
            for sentence_neutral_sentiment in self.sentiment_lexicon.read_neutral_sentence():
                sentence_word = word_tokenize(sentence_neutral_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                self.output_results(sentence_neutral_sentiment, 'neutral', sentence_sentiment_value, sentence_parse_tree)
                self.original.predict_value(sentence_neutral_sentiment, 'neutral')

        except Exception as exception:
            print(exception.args[0])

    def output_results(self, sentence_content: str, original_value: str, sentiment_label_result: List[Tuple[Any, int]],
                       sentence_parse_tree_result: Dict[str, str]) -> None:
        """
        sentiment value result output
        Input:
          self - text file sentence content
          sentence_content - sentence words content
          original_value - sentence original sentiment value
          sentiment_label_result - sentence sentiment pos label result
          sentence_parse_tree_result - sentence sentiment parse tree display
        """
        if len(sentiment_label_result) == 1 and original_value == sentiment_label_result[0][0]:

            # record correct sentiment analysis result with parse tree and pos tagging in Good.txt
            with open("Good.txt", "a+") as writer:
                writer.write('【Sentence for Analysis】\n' + sentence_content + '\r\n')
                writer.write('【Initial Forecast Sentiment Value】\n' + original_value + '\r\n')
                writer.write('【Program Analysis Sentiment Value】\n')
                for i in range(len(sentiment_label_result) - 1):
                    writer.write(sentiment_label_result[i][0] + ', ')
                writer.write(sentiment_label_result[-1][0])
                writer.write('\r\n')
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label_result:
                    writer.write(sentence_parse_tree_result[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n\r\n')
            # record program performance
            if original_value == 'negative':
                self.true_negative += 1
            elif original_value == 'positive':
                self.true_positive += 1
            elif original_value == 'neutral':
                self.true_neutral += 1
        else:

            # record mismatching sentiment analysis result with parse tree and pos tagging in False.txt
            with open("False.txt", "a+") as writer:
                writer.write('【Sentence for Analysis】\n' + sentence_content + '\r\n')
                writer.write('【Initial Forecast Sentiment Value】\n' + original_value + '\r\n')
                writer.write('【Program Analysis Sentiment Value】\n')
                for i in range(len(sentiment_label_result) - 1):
                    writer.write(sentiment_label_result[i][0])
                writer.write('\r\n')
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label_result:
                    writer.write(sentence_parse_tree_result[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n\r\n')
            # record program performance
            for predict_value, cnt in sentiment_label_result:
                if predict_value == 'negative' and predict_value != original_value and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_negative += 1
                elif predict_value == 'positive' and predict_value != original_value and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_positive += 1
                elif predict_value == 'neutral' and predict_value != original_value and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_neutral += 1

    def performance(self) -> None:
        """
        program sentiment performance result view
        Input:
          self - text file sentence content
        """
        print('True Positive\t', self.true_positive, "\tsentences")
        print('True Negative\t', self.true_negative, "\tsentences")
        print('True Neutral\t', self.true_neutral, "\tsentences")
        print('False Positive\t', self.false_positive, "\tsentences")
        print('False Negative\t', self.false_negative, "\tsentences")
        print('False Neutral\t', self.false_neutral, "\tsentences")

    def lexicon_output(self):
        """
        program sentence lexicon result view
        Input:
          self - text file sentence content
        """
        print('Positive Sentences:')
        for sentence in self.sentiment_lexicon.read_positive_sentence():
            print(sentence)
        print()
        print('Negative Sentences:')
        for sentence in self.sentiment_lexicon.read_negative_sentence():
            print(sentence)
        print()
        print('Neutral Sentences:')
        for sentence in self.sentiment_lexicon.read_neutral_sentence():
            print(sentence)


if __name__ == '__main__':

    # grammar file path definition
    feature_grammar = 'grammar.fcfg'
    sentence_parser = SentenceParser(feature_grammar, False, False, False)
    data = SentenceReader()

    # project pipeline definition
    sentiment_pipeline = SentimentPipeline(sentence_parser, data)
    sentiment_pipeline.sentiment_analysis()
    print("Sentence Sentiment Analysis Finish!")
    print()
    print("The Performance of SSAP:")
    sentiment_pipeline.original.performance_result()
    print()
    print("The Performance of Project:")
    sentiment_pipeline.performance()
    print()
    print("Sentence Result:")
    sentiment_pipeline.lexicon_output()
