import marisa_trie
import re

class TrieSearch(marisa_trie.Trie):
    def __init__(self, patterns=None, filepath=None, splitter=' '):
        if filepath:
            super().__init__()
            self.load(filepath)
        else:
            super().__init__(patterns or [])
        self.splitter = splitter

    def search_all_patterns(self, text):
        text_idx = 0
        for line in re.split(r'[\n\r]', text):
            if self.splitter:
                words = re.split(self.splitter, line)
            else:
                words = list(line)
            line_idx = 0
            for i, w in enumerate(words):
                for pattern in self.__search_prefix_patterns(w, words[i + 1:]):
                    yield pattern, text_idx + line_idx
                line_idx += len(w) + len(self.splitter)
            text_idx += line_idx

    def search_longest_patterns(self, text):
        all_patterns = self.search_all_patterns(text)
        check_field = [0] * len(text)
        for pattern, start_idx in sorted(
                all_patterns, key=lambda x: len(x[0]), reverse=True):
            target_field = check_field[start_idx:start_idx + len(pattern)]
            check_sum = sum(target_field)
            if check_sum == 0:
                for i in range(len(pattern)):
                    check_field[start_idx + i] = 1
                yield pattern, start_idx

    def __search_prefix_patterns(self, query, remaining_words):
        if query in self:
            yield query
        if len(remaining_words):
            next_query = self.splitter.join((query, remaining_words[0]))
            gen = self.iterkeys(prefix=next_query)
            try:
                _ = next(gen)
                yield from self.__search_prefix_patterns(
                    next_query, remaining_words[1:])
            except StopIteration:
                pass
