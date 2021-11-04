"""
Project 3 for COMP 6751 Natural Language Analyst
Sentence Analysis Development Program
Be sure to read the project description page for further information about the expected behavior of the program
@author: Haochen Zou (Luke) 40158179
"""
import nltk
from nltk import word_tokenize, load_parser, FeatureEarleyChartParser
from typing import List, Tuple, Dict, Any
from collections import Counter
from collections import defaultdict


class SentenceParser:

    def __init__(self, grammar_file: str, print_parse_tree: bool = False, save_parse_tree: bool = False):
        """
        sentence tree parser initializer
        Input:
          self - text file sentence content
          grammar_file - sentiment analysis grammar file
          print_parse_tree - print parse tree result
          draw_parse_tree - draw parse tree result
          save_parse_tree - save parse tree result
        Output:
          print_result - print parse tree
          draw_result - draw parse tree
          save_result - save parse tree
          tree_number - parse tree serial number
        """
        self.cp = load_parser(grammar_file, trace=0, parser=FeatureEarleyChartParser)
        self.print_result = print_parse_tree
        self.draw_result = save_parse_tree
        self.tree_number = 1

    def parse(self, token: List[str]) -> Tuple[list, Dict[str, List[str]]]:
        """
        parse the sentence
        Input:
          self - text file sentence content
          token - sentence token
        Output:
          associate sentiment values to constituent label
        """
        sentiment_value = []
        parse_tree: Dict[str, List[str]] = defaultdict(list)

        for tree in self.cp.parse(token):
            if self.print_result:
                print(tree)
            if self.draw_result:
                tree.draw_result()

            sentiment_label = tree.label()['SENTIMENT']
            if sentiment_label in ['negative', 'positive', 'neutral']:
                sentiment_value.append(sentiment_label)
                parse_tree[sentiment_label].append(str(tree))
            self.tree_number += 1

        return sentiment_value, parse_tree


class DataReader:
    def __init__(self):
        """
        data reader initializer
        Input:
          self - data file sentence content
        """
        self.sentence_positive = []
        self.sentence_negative = []
        self.sentence_neutral = []

        sentence_positive_file = 'positive.txt'
        sentence_negative_file = 'negative.txt'
        sentence_neutral_file = 'neutral.txt'

        with open(sentence_positive_file, 'r') as reader:
            self.sentence_positive = reader.readlines()
        self.sentence_positive = [sent.rstrip() for sent in self.sentence_positive]

        with open(sentence_negative_file, 'r') as reader:
            self.sentence_negative = reader.readlines()
        self.sentence_negative = [sent.rstrip() for sent in self.sentence_negative]

        with open(sentence_neutral_file, 'r') as reader:
            self.sentence_neutral = reader.readlines()
        self.sentence_neutral = [sent.rstrip() for sent in self.sentence_neutral]

    def read_positive_sentence(self) -> List[str]:
        return self.sentence_positive

    def read_negative_sentence(self) -> List[str]:
        return self.sentence_negative

    def read_neutral_sentence(self) -> List[str]:
        return self.sentence_neutral


def part_of_speech_tagging(word: List[str]) -> List[Tuple[str, str]]:
    """
    part-of-speech tagging function
    Input:
      word: pos word list
    """
    normal_pos_tag = nltk.pos_tag(word[:-1])
    return normal_pos_tag


class Pipeline:
    def __init__(self, earley_parser: SentenceParser, lexicon_data_reader: DataReader):
        """
        pipeline initializer
        Input:
          self - data file sentence content
          earley_parser - sentence earley parser
          lexicon_data_reader - lexicon data reader
        Output:
          sentiment analysis result
        """
        self.data_parser = earley_parser
        self.data_lexicon = lexicon_data_reader

        # initialize the value to compare the result between initial forecast and sentiment analysis
        self.initial_forecast_positive = 0
        self.initial_forecast_negative = 0
        self.initial_forecast_neutral = 0
        self.sentiment_analysis_positive = 0
        self.sentiment_analysis_negative = 0
        self.sentiment_analysis_neutral = 0

    def sentiment_label_parse(self, token_list: List[str]) -> Tuple[List[Tuple[Any, int]], Dict[str, List[str]]]:
        """
        parse the sentence with sentiment label
        Input:
          self - data file sentence content
          token_list - sentiment token list
        Output:
          sentiment analysis result
        """
        sentiments, parse_trees = self.data_parser.parse(token_list[:-1])
        return Counter(sentiments).most_common(), parse_trees

    def sentiment_analysis(self) -> None:
        """
        analysis sentence sentiment value
        Input:
          self - data file sentence content
        Output:
          sentiment analysis result
        """
        try:
            print("Sentence Sentiment Analysis Program\n")
            print("Sentence Sentiment Analysis Start!")

            for positive_sentence in self.data_lexicon.read_positive_sentence():
                sentence_word = word_tokenize(positive_sentence)
                sentence_sentiment, parse_tree = self.sentiment_label_parse(sentence_word)
                self.record_parse_result(positive_sentence, 'positive', sentence_sentiment, parse_tree)

            for negative_sentence in self.data_lexicon.read_negative_sentence():
                sentence_word = word_tokenize(negative_sentence)
                sentence_sentiment, parse_tree = self.sentiment_label_parse(sentence_word)
                self.record_parse_result(negative_sentence, 'negative', sentence_sentiment, parse_tree)

            for neutral_sentence in self.data_lexicon.read_neutral_sentence():
                sentence_word = word_tokenize(neutral_sentence)
                sentence_sentiment, parse_tree = self.sentiment_label_parse(sentence_word)
                self.record_parse_result(neutral_sentence, 'neutral', sentence_sentiment, parse_tree)

        except Exception as ex:
            print(ex.args[0])

        print("Sentence Sentiment Analysis Complete!")

    def record_parse_result(self, sentence: str, sentiment_value: str, sentiment_label: List[Tuple[Any, int]],
                            parse_tree: Dict[str, str]) -> None:
        """
        record the sentence sentiment value analysis result in a txt file
        Input:
          self - data file sentence content
          sentiment_value - sentence sentiment value before analysis
          sentiment_label - sentence sentiment label
          parse_tree - sentence parse tree
        Output:
          sentiment analysis result txt file
        """
        if sentiment_value in [label[0] for label in sentiment_label]:

            with open("Good.txt", "a+") as writer:
                writer.write('【Sentence for Analysis】\n' + sentence + '\r\n')
                writer.write('【Initial Forecast Sentiment Value】\n' + sentiment_value + '\r\n')
                writer.write('【Program Analysis Sentiment Value】\n' + sentiment_label[-1][0] + '\r\n')
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label:
                    writer.write(parse_tree[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n')

            if sentiment_value == 'negative':
                self.initial_forecast_negative += 1
            elif sentiment_value == 'positive':
                self.initial_forecast_positive += 1
            elif sentiment_value == 'neutral':
                self.initial_forecast_neutral += 1
        else:

            with open("False.txt", "a+") as writer:
                writer.write('【Sentence for Analysis】\n' + sentence + '\r\n')
                writer.write('【Initial Forecast Sentiment Value】\n' + sentiment_value + '\r\n')
                writer.write('【Program Analysis Sentiment Value】\n' + sentiment_label[-1][0] + '\r\n')
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label:
                    writer.write(parse_tree[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n')

            if 'negative' in sentiment_label:
                self.sentiment_analysis_negative += 1
            if 'positive' in sentiment_label:
                self.sentiment_analysis_positive += 1
            if 'neutral' in sentiment_label:
                self.sentiment_analysis_neutral += 1


if __name__ == '__main__':

    feature_grammar = 'grammar.fcfg'
    parser = SentenceParser(feature_grammar, False, False)
    data = DataReader()
    result = Pipeline(parser, data)
    result.sentiment_analysis()
