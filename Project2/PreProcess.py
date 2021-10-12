"""
This is the base version which use nltk.pos_tag and nltk.ne_chunk for POS Tagging
and Name Entity Module.
"""

import os
import nltk
import re
from nltk import sent_tokenize, word_tokenize, load_parser, FeatureEarleyChartParser, parse
from typing import List, Tuple, Set
from nltk.draw.tree import TreeView


def part_of_speech_tagging(words: List[str]) -> List[Tuple[str, str]]:
    """
    perform part-of-speech tagging using StanfordPOSTagger
    :param words: a list of words in a sentence
    :param multi_word_name_entities: a set of multi-word name entities
    :return: part-of-speech tag of the sentence
    """
    normal_pos_tag = nltk.pos_tag(words[:-1])   # omit the last period

    return normal_pos_tag


def name_entity_module(word_list: List[str]) -> Tuple[List[str], Set[str]]:
    """
    perform named entity recognition using StanfordNERTagger
    :param word_list: token list
    :return: a token list after merging name entities + a set of name entities
    """
    pos_tag_list = nltk.tag.pos_tag(word_list)  # do POS tagging before chunking
    ne_parse_tree = nltk.ne_chunk(pos_tag_list)
    name_entity: Set[str] = set()
    word_list_merge: List[str] = list()

    for node in ne_parse_tree:
        if isinstance(node, nltk.tree.Tree) and node.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
            ne = ' '.join([word for (word, tag) in node.leaves()])
            name_entity.add(ne)
            word_list_merge.append(ne)
        elif isinstance(node, tuple):
            word_list_merge.append(node[0])
        elif isinstance(node, str):
            word_list_merge.append(node)
    return word_list_merge, name_entity


class Parser:
    def __init__(self, grammar_content: str, print_result: bool = False, save_result: bool = False):
        """
        constructor
        :param grammar_content: grammar file URL
        """
        self.cp = load_parser(grammar_content, trace=0, parser=FeatureEarleyChartParser)
        self.print = print_result
        self.save = save_result
        self.tree_no = 1

    def parse(self, tokens: List[str]) -> None:
        """
        parse sentences in sent and print the parse tree
        :param tokens:
        :param
        """
        for tree in self.cp.parse(tokens):
            print(tree)     # print the tree
            if self.print:
                tree.draw()     # display the tree diagram
            if self.save:
                # save the tree diagram
                TreeView(tree).cframe.print_to_file('results/Tree' + str(self.tree_no) + '_diagram' + '.ps')
                # save the tree text
                with open('results/Tree' + str(self.tree_no) + '_text' + '.txt', "w", encoding='utf-8') as writer:
                    writer.write(str(tree))
            self.tree_no += 1

    def clear_directory(self):
        if self.save:
            # delete all files in /results/.. directory
            direction = 'results/'
            file_list = [f for f in os.listdir(direction)]
            for f in file_list:
                os.remove(os.path.join(direction, f))


class Pipeline:
    def __init__(self, parse_method: Parser, sentence_content: str):
        """
        constructor
        :param parse_method: parser that will be used in pipeline
        """
        self.parser = parse_method
        self.sentence = sentence_content
        self.raw = None

    def read_raw_data(self):
        """
        read raw data from the url
        """
        # check data file validity and read sentence from the file
        sent_content = self.sentence
        if os.path.exists(sent_content):
            with open(sent_content) as file_content:
                self.raw = file_content.read()
        else:
            raise Exception('Error: File ' + sent_content + ' does not exist!')

    def reformat_raw(self) -> str:
        """
        If text in raw data file contains multiple lines, then merge into 1 lines separated by a space.
        :return one-line reformed raw text
        """
        raw_reformat = ""
        for line in self.raw.split('\n'):
            raw_reformat += line.strip() + ' '
        return raw_reformat

    def parse_and_validate(self, token_lists: List[List[str]], pos_tags: List[List[str]]) -> None:
        """
        parse the sentences and print the parse trees
        :param token_lists: a list of token lists of sentences
        """
        self.parser.clear_directory()
        for ts in token_lists:
            self.parser.parse(ts)

    def run_validation(self):
        try:
            # read the text from a local file
            self.read_raw_data()
            # reformat the text
            raw = self.reformat_raw()

            # sentence splitting
            sentence = sent_tokenize(raw)
            print('\nSentences splitting results:')
            print(sentence)

            # tokenization + pos tagging
            pos_tag: List[List[str]] = list()
            token_list: List[List[str]] = list()
            name_entity: Set[str] = set()
            for text_sentence in sentence:
                # word tokenization
                words = word_tokenize(text_sentence)
                # name entity module
                words, name_entity = name_entity_module(words)
                token_list.append(words[:-1])   # omit the last period
                name_entity = name_entity.union(name_entity)
                # part-of-speech tagging
                pos_tag.append(part_of_speech_tagging(words))

            print('\nPart-of-speech tagging results:')
            print(pos_tag)

            print('\nName entities:')
            print(name_entity)

            # run the Earley parser written in context-free grammar to validate data
            print('\nParsing results:')
            self.parse_and_validate(token_list, pos_tag)

            with open("text.txt", "r") as f:
                data = f.read()
            text_sentence = data
            sentence_without_punctuation = re.sub(r'[^\w\s]','',text_sentence)
            tokens = sentence_without_punctuation.split()
            cp = parse.load_parser('grammar.fcfg', trace=1)

            print('\nEarley parse process:')
            trees = cp.parse(tokens)
            print(trees)

        except Exception as ex:
            print(ex.args[0])


if __name__ == '__main__':

    # define an Earley parser and load the grammar rules
    print("Please enter the text you want to parse:")
    text = input()
    file_name = 'text.txt'

    with open('text.txt', 'w') as file:
        file.write(text)
    data_file = 'text.txt'

    print("\nOptions to the results: save/print")
    input_save = input('Do you want to save the parse tree? ')
    input_pprint = input('Do you want to print the parse tree? ')
    pprint = False
    save = False
    if input_pprint.upper() == 'Y' or input_pprint.lower() == 'yes':
        pprint = True
    if input_save.upper() == 'Y' or input_save.lower() == 'yes':
        save = True
    grammar_file_url = 'grammar.fcfg'
    parser = Parser(grammar_file_url, pprint, save)

    # run pipeline to validate the data
    pipeline = Pipeline(parser, data_file)
    pipeline.run_validation()

