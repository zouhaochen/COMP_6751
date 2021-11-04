"""
Project 2 for COMP 6751 Natural Language Analyst
Context Free Grammar Development Program
Be sure to read the project description page for further information about the expected behavior of the program
@author: Haochen Zou
"""

import os
import re
import nltk
from nltk import sent_tokenize, word_tokenize, load_parser, FeatureEarleyChartParser, parse
from typing import List, Tuple, Set
from nltk.draw.tree import TreeView


def pos_tagging(words: List[str]) -> List[Tuple[str, str]]:
    """
    Inputs:
      words - words content of text file
    Output:
      part-of-speech tagging results
    """
    normal_pos_tag = nltk.pos_tag(words[:-1])   # omit the last period

    return normal_pos_tag


def name_entity_module(word_list: List[str]) -> Tuple[List[str], Set[str]]:
    """
    Inputs:
      word_list - word token list of text file
    Output:
      token list with name entities
    """
    pos_tag_list = nltk.tag.pos_tag(word_list)
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

    # initialize parser structure information
    def __init__(self, grammar_content: str, print_result: bool = False, save_result: bool = False):
        self.cp = load_parser(grammar_content, trace=0, parser=FeatureEarleyChartParser)
        self.print = print_result
        self.save = save_result
        self.tree_no = 1

    def parse(self, tokens: List[str]) -> None:
        """
        Inputs:
          self - text file sentence content
          tokens - token for parse
        Output:
          parse tree result print
          save the tree in text and diagram format
        """
        for tree in self.cp.parse(tokens):
            print(tree)
            if self.print:
                tree.draw_result()
            if self.save:
                TreeView(tree).cframe.print_to_file('results/Tree' + str(self.tree_no) + '_diagram' + '.ps')
                with open('results/Tree' + str(self.tree_no) + '_text' + '.txt', "w", encoding='utf-8') as writer:
                    writer.write(str(tree))
            self.tree_no += 1

    def save_result(self):
        """
        Inputs:
          self - text file sentence content
        Output:
          save result in direction
        """
        if self.save:
            direction = 'results/'
            file_list = [f for f in os.listdir(direction)]
            for f in file_list:
                os.remove(os.path.join(direction, f))


class Pipeline:

    # pipeline constructor
    def __init__(self, parse_method: Parser, sentence_content: str):
        self.parser = parse_method
        self.sentence = sentence_content
        self.raw = None

    def text_read(self):
        """
        Inputs:
          self - text file sentence content
        Output:
          file exist or not, display content if exist
        """
        sent_content = self.sentence
        if os.path.exists(sent_content):
            with open(sent_content) as file_content:
                self.raw = file_content.read()
        else:
            raise Exception('Error: File ' + sent_content + ' does not exist!')

    def text_reformation(self) -> str:
        """
        Inputs:
          self - text file sentence content
        Output:
          reformat text file
        """
        text_reformat = ""
        for line in self.raw.split('\n'):
            text_reformat += line.strip() + ' '
        return text_reformat

    def parse_print_text(self, token_lists: List[List[str]]) -> None:
        """
        Inputs:
          self - text file sentence content
          token_lists - a list of token lists of sentences
        Output:
          parse tree
        """
        self.parser.save_result()
        for ts in token_lists:
            self.parser.parse(ts)

    def text_preprocess(self):
        try:
            self.text_read()
            raw = self.text_reformation()

            # sentence splitting
            sentence = sent_tokenize(raw)
            print('\n【Sentences Splitting】')
            print(sentence)

            # tokenize, name entity and part-of-speech tagging
            pos_tag: List[List[str]] = list()
            token_list: List[List[str]] = list()
            name_entity: Set[str] = set()
            for text_sentence in sentence:
                words = word_tokenize(text_sentence)
                words, name_entity = name_entity_module(words)
                token_list.append(words[:-1])
                name_entity = name_entity.union(name_entity)
                pos_tag.append(pos_tagging(words))

            # part-of-speech tagging result
            print('\n【POS Tagging】')
            print(pos_tag)

            # name entities result
            print('\n【Name Entities】')
            print(name_entity)

            # Earley parsing result
            print('\n【Earley Parsing】')
            self.parse_print_text(token_list)

            # Write and read text file content form a txt file
            with open("text.txt", "r") as f:
                data = f.read()
            text_sentence = data
            sentence_without_punctuation = re.sub(r'[^\w\s]','',text_sentence)
            tokens = sentence_without_punctuation.split()
            cp = parse.load_parser('grammar.fcfg', trace=1)

            # Earley parsing process result
            print('\n【Earley Parse Process】')
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

    # pipeline initializer
    pipeline = Pipeline(parser, data_file)
    pipeline.text_preprocess()

