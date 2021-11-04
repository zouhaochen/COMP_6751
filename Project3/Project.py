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
          print_result - print parse tree
          draw_result - draw parse tree
          save_result - save parse tree
          tree_number - parse tree serial number
        """
        self.cp = load_parser(grammar_file, trace=0, parser=FeatureEarleyChartParser)
        self.print_result = print_parse_tree
        self.draw_result = draw_parse_tree
        self.save_result = save_parse_tree
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

    def read_negative_sentences(self) -> List[str]:
        return self.sentence_negative

    def read_positive_sentences(self) -> List[str]:
        return self.sentence_positive

    def read_neutral_sentences(self) -> List[str]:
        return self.sentence_neutral


class Pipeline:
    def __init__(self, parser: SentenceParser, lexica: DataReader):
        """
        constructor
        :param parser: Earley parser
        :param lexica: Lexica data loader
        """
        self.parser = parser
        self.lexica = lexica
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.true_neutral = 0
        self.false_neutral = 0

    def part_of_speech_tagging(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        perform part-of-speech tagging
        :param words: a list of words
        """
        normal_pos_tag = nltk.pos_tag(words[:-1])  # omit the last period
        return normal_pos_tag

    def parse_and_sentify(self, token_list: List[str]) -> Tuple[List[Tuple[Any, int]], Dict[str, str]]:
        """
        parse the sentence
        :param token_list: tokens of a sentence
        :return return the most possible sentiment
        """
        # retrieve sentiment labels of all possible parse trees
        sentiments, parse_trees = self.parser.parse(token_list[:-1])  # omit the last period
        # return the most probable sentiment
        return Counter(sentiments).most_common(), parse_trees

    def run_pipeline(self) -> None:
        """
        run the sentiment pipeline
        """
        try:

            # tokenization + pos tagging
            # positive sentences
            for pos_sent in self.lexica.read_positive_sentences():
                words = word_tokenize(pos_sent)
                senti, trees = self.parse_and_sentify(words)
                # write the sentencee and the ground-truth and the prediction to a result file
                self.output_results(pos_sent, 'positive', senti, trees)
                print("Positive Sentence Sentiment Value Analysis Complete")
            # negative sentences
            for neg_sent in self.lexica.read_negative_sentences():
                words = word_tokenize(neg_sent)
                senti, trees = self.parse_and_sentify(words)
                # write the ground-truth and prediction to a result file
                self.output_results(neg_sent, 'negative', senti, trees)
                print("Negative Sentence Sentiment Value Analysis Complete")
            # neutral sentences
            for neu_sent in self.lexica.read_neutral_sentences():
                words = word_tokenize(neu_sent)
                senti, trees = self.parse_and_sentify(words)
                # write the ground-truth and prediction to a result file
                self.output_results(neu_sent, 'neutral', senti, trees)
                print("Neutral Sentence Sentiment Value Analysis Complete")
        except Exception as ex:
            print(ex.args[0])

    def output_results(self, sentence: str, ground_truth: str, labels: List[Tuple[Any, int]],
                       trees: Dict[str, str]) -> None:
        """
        write the result with the following format
            sentence1
            ground_truth label  prediction label
            sentence 2
            ground_truth label  prediction label
            ...
        :param sentence: sentence
        :param ground_truth: ground-truth label
        :param label: prediction label
        """
        if ground_truth in [label[0] for label in labels]:
            # write the sentence and the ground_truth and label to Good.txt
            with open("Good.txt", "a+") as writer:
                # write the input sentence
                writer.write(sentence + '\r\n')
                # write the ground-truth label
                writer.write(ground_truth + '\t|\t')
                # write the prediction labels
                writer.write('[')
                for i in range(len(labels) - 1):
                    writer.write(labels[i][0] + ', ')
                writer.write(labels[-1][0] + ']')
                writer.write('\r\n\r\n')
                # write the first tree with prediction label that has the most votes
                for label in labels:
                    writer.write(trees[label[0]][0])
                    writer.write('\r\n')
                writer.write('-------------------------------------------------------------------\r\n')
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
                writer.write(sentence + '\r\n')
                # write the ground-truth label
                writer.write(ground_truth + '\t|\t')
                # write the prediction labels
                writer.write('[')
                for i in range(len(labels) - 1):
                    writer.write(labels[i][0] + ', ')
                writer.write(labels[-1][0] + ']')
                writer.write('\r\n\r\n')
                # write the first tree with prediction label that has the most votes
                for label in labels:
                    writer.write(trees[label[0]][0])
                    writer.write('\r\n')
                writer.write('-------------------------------------------------------------------\r\n')
            # record performance
            if 'negative' in labels:
                self.false_negative += 1
            if 'positive' in labels:
                self.false_positive += 1
            if 'neutral' in labels:
                self.false_neutral += 1


if __name__ == '__main__':
    # define the parser
    grammar_url_s = 'grammar.fcfg'
    parser = SentenceParser(grammar_url_s, False, False, False)
    # load the data from nltk
    data = DataReader()

    # define and run pipeline
    sp = Pipeline(parser, data)
    # sp.print_lexica()
    sp.run_pipeline()
