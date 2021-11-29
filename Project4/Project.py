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
        constructor
        :param grammar_file: grammar file URL
        :param print: whether print the tree on console
        :param draw_parse_tree: whether draw the parse tree on console
        :param save_parse_tree: whether save the parse tree on drive
        """
        self.parser_init = load_parser(grammar_file, trace=0, parser=FeatureEarleyChartParser)
        self.print_result = print_parse_tree
        self.draw_result = draw_parse_tree
        self.save_result = save_parse_tree
        self.tree_number = 1

    def parse(self, token: List[str]) -> Tuple[list, Dict[str, List[str]]]:
        """
        parse sentences in sent and print the parse tree
        :param token: tokens of a sentence
        :return all possible sentiment labels
        """
        sentiment_value = []
        parse_tree: Dict[str, List[str]] = defaultdict(list)

        # parse the sentence where S is the root
        for tree in self.parser_init.parse(token):
            if self.print_result:
                print(tree)
            if self.draw_result:
                tree.draw_result()

            # append the root's SENTI attribute value to the list
            sentiment_label = tree.label()['SENTIMENT']
            if sentiment_label in ['negative', 'positive', 'neutral']:
                sentiment_value.append(sentiment_label)
                parse_tree[sentiment_label].append(str(tree))
            self.tree_number += 1

        if len(sentiment_value) == 0:
            sentiment_value.append('unknown')
            parse_tree['unknown'].append('(unknown)')

        return sentiment_value, parse_tree


class DataReader:
    def __init__(self):
        """
        constructor
        """
        self.sentence_positive = []
        self.sentence_negative = []
        self.sentence_neutral = []

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
        return self.sentence_negative

    def read_positive_sentence(self) -> List[str]:
        return self.sentence_positive

    def read_neutral_sentence(self) -> List[str]:
        return self.sentence_neutral


class AfinnAndSsap:
    def __init__(self):
        # AFINN-111 is as of June 2011 the most recent version of AFINN
        afinn_file_name = 'AFINN-111.txt'
        # afinn = dict(map(lambda w, s: (w, int(s)), [ws.strip().split('\t') for ws in open(filenameAFINN)])) # python 2
        self.afinn = dict(
            map(lambda ws: (ws[0], int(ws[1])), [ws.strip().split('\t') for ws in open(afinn_file_name)]))  # python 3
        # Word splitter pattern
        self.pattern_split = re.compile(r"\W+")

        self.polarity = []
        # performance
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.true_neutral = 0
        self.false_neutral = 0

    def sentiment(self, text):
        """
        Returns a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative valence.
        """
        sentence_word = self.pattern_split.split(text.lower())
        # sentiments = map(lambda word: afinn.get(word, 0), words)    # python 2
        word_sentiment = list(map(lambda word: self.afinn.get(word, 0), sentence_word))  # python 3
        if word_sentiment:
            # How should you weight the individual word sentiments?
            # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
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

    def predict_value(self, text: str, ground_truth: str):
        sentiment_score = self.sentiment(text)
        if ground_truth == 'positive':
            if sentiment_score > 0:
                self.true_positive += 1
            elif sentiment_score == 0:
                self.false_neutral += 1
            elif sentiment_score < 0:
                self.false_negative += 1
        elif ground_truth == 'neutral':
            if sentiment_score > 0:
                self.false_positive += 1
            elif sentiment_score == 0:
                self.true_neutral += 1
            elif sentiment_score < 0:
                self.false_negative += 1
        elif ground_truth == 'negative':
            if sentiment_score > 0:
                self.false_positive += 1
            elif sentiment_score == 0:
                self.false_neutral += 1
            elif sentiment_score < 0:
                self.true_negative += 1

        self.polarity.append(sentiment_score)

    def performance_result(self):
        print('True Positive =', self.true_positive)
        print('True Negative =', self.true_negative)
        print('True Neutral =', self.true_neutral)
        print('False Positive =', self.false_positive)
        print('False Negative =', self.false_negative)
        print('False Neutral =', self.false_neutral)


def part_of_speech_tagging(words: List[str]) -> List[Tuple[str, str]]:
    """
    perform part-of-speech tagging
    :param words: a list of words
    """
    normal_pos_tag = nltk.pos_tag(words[:-1])  # omit the last period
    return normal_pos_tag


class SentimentPipeline:
    def __init__(self, earley_parser: SentenceParser, sentence_lexicon: DataReader):
        """
        constructor
        :param earley_parser: Earley parser
        :param sentence_lexicon: Lexica data loader
        """
        self.sentiment_parser = earley_parser
        self.sentiment_lexicon = sentence_lexicon
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.true_neutral = 0
        self.false_neutral = 0
        self.baseline = AfinnAndSsap()

    def sentence_sentiment_parse(self, token_list: List[str]) -> Tuple[List[Tuple[Any, int]], Dict[str, str]]:
        """
        parse the sentence
        :param token_list: tokens of a sentence
        :return return the most possible sentiment
        """
        # retrieve sentiment labels of all possible parse trees
        sentiment_label, parse_tree = self.sentiment_parser.parse(token_list[:-1])  # omit the last period
        # return the most probable sentiment
        return Counter(sentiment_label).most_common(), parse_tree

    def sentiment_analysis(self) -> None:
        """
        run the sentiment pipeline
        """
        try:

            # tokenization + pos tagging
            # positive sentences
            print("Sentence Sentiment Analysis Program\n")
            print("Sentence Sentiment Analysis Start!")
            for sentence_positive_sentiment in self.sentiment_lexicon.read_positive_sentence():
                sentence_word = word_tokenize(sentence_positive_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                # write the sentencee and the ground-truth and the prediction to a result file
                self.output_results(sentence_positive_sentiment, 'positive', sentence_sentiment_value, sentence_parse_tree)
                # run ssap baseline
                self.baseline.predict_value(sentence_positive_sentiment, 'positive')
            # negative sentences
            for sentence_negative_sentiment in self.sentiment_lexicon.read_negative_sentence():
                sentence_word = word_tokenize(sentence_negative_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                # write the ground-truth and prediction to a result file
                self.output_results(sentence_negative_sentiment, 'negative', sentence_sentiment_value, sentence_parse_tree)
                # run ssap baseline
                self.baseline.predict_value(sentence_negative_sentiment, 'negative')
            # neutral sentences
            for sentence_neutral_sentiment in self.sentiment_lexicon.read_neutral_sentence():
                sentence_word = word_tokenize(sentence_neutral_sentiment)
                sentence_sentiment_value, sentence_parse_tree = self.sentence_sentiment_parse(sentence_word)
                # write the ground-truth and prediction to a result file
                self.output_results(sentence_neutral_sentiment, 'neutral', sentence_sentiment_value, sentence_parse_tree)
                # run ssap baseline
                self.baseline.predict_value(sentence_neutral_sentiment, 'neutral')
        except Exception as exception:
            print(exception.args[0])

    def output_results(self, sentence_content: str, ground_truth: str, sentiment_label_result: List[Tuple[Any, int]],
                       sentence_parse_tree_result: Dict[str, str]) -> None:
        """
        write the result with the following format
            sentence1
            ground_truth label  prediction label
            sentence 2
            ground_truth label  prediction label
            ...
        :param sentence_parse_tree_result: 
        :param sentiment_label_result: 
        :param sentence_content: sentence
        :param ground_truth: ground-truth label
        :param label: prediction label
        """
        # if ground_truth in [label[0] for label in labels]:
        if len(sentiment_label_result) == 1 and ground_truth == sentiment_label_result[0][0]:
            # write the sentence and the ground_truth and label to Good.txt
            with open("Good.txt", "a+") as writer:
                # write the input sentence
                writer.write('【Sentence for Analysis】\n' + sentence_content + '\r\n')
                # write the ground-truth label
                writer.write('【Initial Forecast Sentiment Value】\n' + ground_truth + '\r\n')
                # write the prediction labels
                writer.write('【Program Analysis Sentiment Value】\n')
                for i in range(len(sentiment_label_result) - 1):
                    writer.write(sentiment_label_result[i][0] + ', ')
                writer.write(sentiment_label_result[-1][0])
                writer.write('\r\n')
                # write the first tree with prediction label that has the most votes
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label_result:
                    writer.write(sentence_parse_tree_result[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n\r\n')
            # record performance
            if ground_truth == 'negative':
                self.true_negative += 1
            elif ground_truth == 'positive':
                self.true_positive += 1
            elif ground_truth == 'neutral':
                self.true_neutral += 1
        else:
            # write the sentence and the ground_truth and label to False.txt
            with open("False.txt", "a+") as writer:
                # write the input sentence
                writer.write('【Sentence for Analysis】\n' + sentence_content + '\r\n')
                # write the ground-truth label
                writer.write('【Initial Forecast Sentiment Value】\n' + ground_truth + '\r\n')
                # write the prediction labels
                writer.write('【Program Analysis Sentiment Value】\n')
                for i in range(len(sentiment_label_result) - 1):
                    writer.write(sentiment_label_result[i][0] + ', ')
                writer.write(sentiment_label_result[-1][0] + ']')
                writer.write('\r\n')
                # write the first tree with prediction label that has the most votes
                writer.write('【Earley Parse Result】\r\n')
                for label in sentiment_label_result:
                    writer.write(sentence_parse_tree_result[label[0]][0])
                    writer.write('\r\n')
                writer.write('\r\n\r\n')
            # record performance
            for predict_value, cnt in sentiment_label_result:
                if predict_value == 'negative' and predict_value != ground_truth and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_negative += 1
                elif predict_value == 'positive' and predict_value != ground_truth and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_positive += 1
                elif predict_value == 'neutral' and predict_value != ground_truth and predict_value in [label[0] for label in sentiment_label_result]:
                    self.false_neutral += 1

    def performance(self) -> None:
        # recall = self.true_positive / (self.true_positive + self.false_negative)
        # precision = self.true_positive / (self.true_positive + self.false_positive)
        # f1_score = (precision * recall) / (precision + recall)
        print('True Positive =', self.true_positive)
        print('True Negative =', self.true_negative)
        print('True Neutral =', self.true_neutral)
        print('False Positive =', self.false_positive)
        print('False Negative =', self.false_negative)
        print('False Neutral =', self.false_neutral)
        # print('Precision =', precision)
        # print('Recall =', recall)
        # print('F1 measure =', f1_score)

    def lexicon_output(self):
        print('positive sentences:')
        for sentence in self.sentiment_lexicon.read_positive_sentence():
            print(sentence)
        print()
        print('negative sentences:')
        for sentence in self.sentiment_lexicon.read_negative_sentence():
            print(sentence)


if __name__ == '__main__':
    # define the parser
    feature_grammar = 'grammar.fcfg'
    sentence_parser = SentenceParser(feature_grammar, False, False, False)
    # load the data from nltk
    data = DataReader()

    # define and run pipeline
    sentiment_pipeline = SentimentPipeline(sentence_parser, data)
    sentiment_pipeline.sentiment_analysis()
    print("Sentence Sentiment Analysis Finish!")

    print()
    print("The Performance of SSAP:")
    sentiment_pipeline.baseline.performance_result()
    print()
    print("The Performance of Project:")
    sentiment_pipeline.performance()
