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


class Parser:
    def __init__(self, grammar_url: str, print_result: bool = False, save_result: bool = False):
        """
        constructor
        :param grammar_url: grammar file URL
        """
        self.cp = load_parser(grammar_url, trace=0, parser=FeatureEarleyChartParser)
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
                TreeView(tree)._cframe.print_to_file('results/Tree' + str(self.tree_no) + '_diagram' + '.ps')
                # save the tree text
                with open('results/Tree' + str(self.tree_no) + '_text' + '.txt', "w", encoding='utf-8') as writer:
                    writer.write(str(tree))
            self.tree_no += 1

    def clear_directory(self):
        if self.save:
            # delete all files in /results/.. directory
            dir = 'results/'
            filelist = [f for f in os.listdir(dir)]
            for f in filelist:
                os.remove(os.path.join(dir, f))


class Pipeline:
    def __init__(self, parser: Parser, sent_url: str):
        """
        constructor
        :param parser: parser that will be used in pipeline
        """
        self.parser = parser
        self.sent_url = sent_url
        self.raw = None

    def read_raw_data(self):
        """
        read raw data from the url
        """
        # check data file validity and read sentence from the file
        sent_url = self.sent_url
        if os.path.exists(sent_url):
            with open(sent_url) as file:
                self.raw = file.read()
        else:
            raise Exception('Error: ' + sent_url + ' does not exist on local.')

    def reformat_raw(self) -> str:
        """
        If text in raw data file contains multiple lines, then merge into 1 lines separated by a space.
        :return one-line reformed raw text
        """
        reformed_raw = ""
        for line in self.raw.split('\n'):
            reformed_raw += line.strip() + ' '
        return reformed_raw

    def part_of_speech_tagging(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        perform part-of-speech tagging using StanfordPOSTagger
        :param words: a list of words in a sentence
        :param multi_word_name_entities: a set of multi-word name entities
        :return: part-of-speech tag of the sentence
        """
        normal_pos_tag = nltk.pos_tag(words[:-1])   # omit the last period

        return normal_pos_tag

    def name_entity_module(self, word_list: List[str]) -> Tuple[List[str], Set[str]]:
        """
        perform named entity recognition using StanfordNERTagger
        :param word_list: token list
        :return: a token list after merging name entities + a set of name entities
        """
        pos_tag_list = nltk.tag.pos_tag(word_list)  # do POS tagging before chunking
        ne_parse_tree = nltk.ne_chunk(pos_tag_list)
        name_entities: Set[str] = set()
        word_list_merged: List[str] = list()

        for node in ne_parse_tree:
            if isinstance(node, nltk.tree.Tree) and node.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                ne = ' '.join([word for (word, tag) in node.leaves()])
                name_entities.add(ne)
                word_list_merged.append(ne)
            elif isinstance(node, tuple):
                word_list_merged.append(node[0])
            elif isinstance(node, str):
                word_list_merged.append(node)
        return word_list_merged, name_entities

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
            sents = sent_tokenize(raw)
            print('Sentences splitting results:')
            print(sents)
            print('---------------------------------------------')

            # tokenization + pos tagging
            pos_tags: List[List[str]] = list()
            token_lists: List[List[str]] = list()
            name_entities: Set[str] = set()
            for sent in sents:
                # word tokenization
                words = word_tokenize(sent)
                # name entity module
                words, name_entity = self.name_entity_module(words)
                token_lists.append(words[:-1])   # omit the last period
                name_entities = name_entities.union(name_entity)
                # part-of-speech tagging
                pos_tags.append(self.part_of_speech_tagging(words))
            print('Part-of-speech tagging results:')
            print(pos_tags)
            print('Name entities:')
            print(name_entities)
            print('---------------------------------------------')

            # run the Earley parser written in context-free grammar to validate data
            print('Parsing results:')
            self.parse_and_validate(token_lists, pos_tags)
            print('---------------------------------------------')

            with open("text.txt", "r") as f:
                data = f.read()
            sent = data
            out = re.sub(r'[^\w\s]','',sent)
            tokens = out.split()
            cp = parse.load_parser('grammar/grammar.fcfg', trace=1)
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

    print("Options to the results: save/print")
    input_save = input('Do you want to save the parse tree? ')
    input_pprint = input('Do you want to print the parse tree? ')
    pprint = False
    save = False
    if input_pprint.upper() == 'Y' or input_pprint.lower() == 'yes':
        pprint = True
    if input_save.upper() == 'Y' or input_save.lower() == 'yes':
        save = True
    grammar_file_url = 'grammar/grammar.fcfg'
    parser = Parser(grammar_file_url, pprint, save)

    # run pipeline to validate the data
    pipeline = Pipeline(parser, data_file)
    pipeline.run_validation()

