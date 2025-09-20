from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
from bitarray import bitarray
from itertools import pairwise, count
from heapq import heapify, heappush, heappop
from time import time


class TokenProbability(NamedTuple):
    text: str
    prob: float

    def __repr__(self):
        return f"({self.text}: {self.prob})"


class AlternatePath(NamedTuple):
    start_index: int
    text: str

    def __repr__(self):
        return f"({self.start_index}: {self.text})"


class PredictionModel(ABC):
    """Predicts the probability distribution of the next token given an input string.

    Essentially just a wrapper that serves to guarantee the output is sorted in ascending order of probability.
    """

    @abstractmethod
    def _predict_next(self, text: str) -> list[TokenProbability]: ...

    @abstractmethod
    def _predict_start(self) -> list[TokenProbability]: ...

    def predict(self, text: str) -> list[TokenProbability]:
        if text == "":
            distribution = self._predict_start()
        else:
            distribution = self._predict_next(text)
        assert all(a[-1::-1] <= b[-1::-1] for a, b in pairwise(distribution))
        return distribution


class Node(ABC):
    @abstractmethod
    def bits_to_token(self, bits: bitarray, index: int) -> tuple[str, int]: ...

    @abstractmethod
    def token_to_bits(self, token: str) -> Optional[bitarray]: ...


class LeafNode(Node):
    def __init__(self, token: str):
        self.token = token

    def bits_to_token(self, bits: bitarray, index: int) -> tuple[str, int]:
        return self.token, index

    def token_to_bits(self, token: str) -> Optional[bitarray]:
        if token == self.token:
            return bitarray()
        return None

    def __repr__(self):
        return self.token


class InnerNode(Node):
    def __init__(self, zero_branch: Node, one_branch: Node):
        self.zero_branch = zero_branch
        self.one_branch = one_branch

    def bits_to_token(self, bits: bitarray, index: int) -> tuple[str, int]:
        if index < len(bits):
            next_bit = bits[index]
        else:
            next_bit = 0

        return (self.zero_branch, self.one_branch)[next_bit].bits_to_token(
            bits, index + 1
        )

    def token_to_bits(self, token: str) -> Optional[bitarray]:
        for branch, bit in [
            (self.zero_branch, bitarray("0")),
            (self.one_branch, bitarray("1")),
        ]:
            branch_result = branch.token_to_bits(token)
            if branch_result is not None:
                return bit + branch_result
        return None

    def __repr__(self):
        return f"({repr(self.zero_branch)} {repr(self.one_branch)})"


class WontHappenInPracticeException(Exception):
    """Indicates an edge case that I don’t expect to happen in real usage.

    Some of my test cases are much simpler than any natural-language case and thus result in some edge cases that would not actually show up in practice, in particular a probability distribution with fewer than two tokens. If a random test produces this exception, it will just skip it and generate new random tests until it gets one that works.
    """


def generate_huffman_code(distribution: list[TokenProbability]) -> InnerNode:
    if len(distribution) == 0:
        raise WontHappenInPracticeException

    counter = count()

    code_q: list[tuple[float, int, Node]] = [
        (token.prob, next(counter), LeafNode(token.text)) for token in distribution
    ]
    heapify(code_q)

    while len(code_q) > 1:
        min_0_prob, _, min_0_tree = heappop(code_q)
        min_1_prob, _, min_1_tree = heappop(code_q)
        combined = (
            min_0_prob + min_1_prob,
            next(counter),
            InnerNode(min_0_tree, min_1_tree),
        )
        heappush(code_q, combined)

    result = code_q[0][2]
    if not isinstance(result, InnerNode):
        raise WontHappenInPracticeException
    return result


class Timer:
    def __init__(self):
        self.times: dict[str, float] = {}

    def start(self, label):
        self.times[label] = time()

    def stop(self, label):
        self.times[label] = time() - self.times[label]

    def __repr__(self):
        return str(self.times)


class Translator(ABC):
    def __init__(
        self,
        prediction_model: PredictionModel,
        expansion_threshold=0.5,
        expansion_tolerance=0.1,
    ):
        self.prediction_model = prediction_model
        self.expansion_threshold = expansion_threshold
        self.expansion_tolerance = expansion_tolerance
        self.timers: list[Timer] = []

        self.alternate_paths: list[AlternatePath] = []

    @property
    @abstractmethod
    def processed_text(self) -> str: ...

    @property
    @abstractmethod
    def processed_text_length(self) -> int: ...

    def predict_tokens(self) -> list[TokenProbability]:
        """Generate the probability distribution for the next token.

        Starts by applying the model once; then, if any token has a probability greater than expansion_threshold, replaces that one token with 2-grams whose total probability is within expansion_tolerance of the actual value. Repeat this process with longer n-grams if necessary until no token/n-gram has a probablity greater than expansion_threshold.

        Returns:
            list[TokenProbability]: _description_
        """

        self.timers[-1].start("generate distribution")

        distribution = self.prediction_model.predict(self.processed_text)

        ngrams_to_expand: list[TokenProbability] = []
        for tp in reversed(distribution):
            if tp.prob > self.expansion_threshold:
                ngrams_to_expand.append(tp)
                distribution.pop()

        while ngrams_to_expand:
            next_ngrams_to_expand: list[TokenProbability] = []

            while ngrams_to_expand:
                ngram_to_expand = ngrams_to_expand.pop()

                next_token_dist = self.prediction_model.predict(
                    self.processed_text + ngram_to_expand.text
                )
                expanded_ngram: list[TokenProbability] = []
                cumulative_prob = 0
                next_token_index = len(next_token_dist) - 1
                while cumulative_prob < ngram_to_expand.prob - self.expansion_tolerance:
                    next_token = next_token_dist[next_token_index]
                    adjusted_prob = ngram_to_expand.prob * next_token.prob
                    expanded_ngram.append(
                        TokenProbability(
                            ngram_to_expand.text + next_token.text,
                            adjusted_prob,
                        )
                    )
                    cumulative_prob += adjusted_prob
                    next_token_index -= 1

                remaining_prob = ngram_to_expand.prob - cumulative_prob
                assert remaining_prob < self.expansion_threshold
                expanded_ngram.append(
                    TokenProbability(ngram_to_expand.text, remaining_prob)
                )

                under_threshold, over_threshold = [], []
                for n_plus_1_gram in expanded_ngram:
                    (under_threshold, over_threshold)[
                        n_plus_1_gram.prob > self.expansion_threshold
                    ].append(n_plus_1_gram)

                for token_to_add in under_threshold:
                    distribution = [
                        token
                        for token in distribution
                        if token.text != token_to_add.text
                    ]
                distribution += under_threshold
                next_ngrams_to_expand += over_threshold

            ngrams_to_expand = next_ngrams_to_expand

        self.timers[-1].stop("generate distribution")

        return distribution

    def filter_tokens(
        self, tokens: list[TokenProbability], times=None
    ) -> list[TokenProbability]:
        """Filter out tokens that would create ambiguity when reading.

        Takes into consideration any other choices that could have been made in previous steps (stored in self.alternate_paths); if any of the tokens considered would result in one of those paths being chosen instead (by the greedy tokenizer), removes that token. Also filters self.alternate_paths to remove ones that are no longer relevant.

        Args:
            tokens (list[TokenProbability]): the distribution returned by the model

        Returns:
            list[TokenProbability]: filtered token distribution
        """

        self.timers[-1].start("filter tokens")

        filtered_tokens = tokens.copy()
        filtered_alternate_paths: list[AlternatePath] = []

        for alternate_path in self.alternate_paths:
            chosen_path_from_this_point = self.processed_text[
                alternate_path.start_index :
            ]
            if not alternate_path.text.startswith(chosen_path_from_this_point):
                continue
            filtered_alternate_paths.append(alternate_path)

            filtered_tokens = [
                token
                for token in filtered_tokens
                if not (chosen_path_from_this_point + token.text).startswith(
                    alternate_path.text
                )
            ]

        self.alternate_paths = filtered_alternate_paths
        self.timers[-1].stop("filter tokens")
        return filtered_tokens

    def generate_distribution(
        self,
    ) -> list[TokenProbability]:
        return self.filter_tokens(self.predict_tokens())

    def record_alternate_paths(
        self, distribution: list[TokenProbability], chosen_token: str
    ):
        for token in distribution:
            if token.text != chosen_token:
                self.alternate_paths.append(
                    AlternatePath(self.processed_text_length, token.text)
                )


class BitsToText(Translator):
    def __init__(self, prediction_model: PredictionModel, bits: bitarray, **kwargs):
        super().__init__(prediction_model, **kwargs)
        self.bits = bits
        self.text = ""

    @property
    def processed_text(self):
        return self.text

    @property
    def processed_text_length(self):
        return len(self.text)

    def translate(self, progress=None) -> str:
        bit_index = 0

        while bit_index < len(self.bits):
            self.timers.append(Timer())

            distribution = self.generate_distribution()
            self.timers[-1].start("generate huffman code")
            code = generate_huffman_code(distribution)
            self.timers[-1].stop("generate huffman code")
            old_bit_index = bit_index
            next_token, bit_index = code.bits_to_token(self.bits, bit_index)
            assert isinstance(next_token, str)
            self.record_alternate_paths(distribution, next_token)
            self.text += next_token

            if progress == "tokens":
                print(next_token, self.bits[old_bit_index:bit_index].to01())
            elif progress == "text":
                print(next_token, end="")

        return self.text


class TextToBits(Translator):
    def __init__(self, prediction_model: PredictionModel, text: str, **kwargs):
        super().__init__(prediction_model, **kwargs)
        self.text = text
        self.text_index = 0

    @property
    def processed_text(self):
        return self.text[: self.text_index]

    @property
    def processed_text_length(self):
        return self.text_index

    def translate(self, progress=None) -> bitarray:
        bits = bitarray()

        while self.text_index < len(self.text):
            self.timers.append(Timer())

            distribution = self.generate_distribution()

            next_token = ""
            for token in distribution:
                if self.text.startswith(token.text, self.text_index) and len(
                    token.text
                ) > len(next_token):
                    next_token = token.text

            self.timers[-1].start("generate huffman code")
            code = generate_huffman_code(distribution)
            self.timers[-1].stop("generate huffman code")
            next_bits = code.token_to_bits(next_token)
            assert isinstance(next_bits, bitarray)
            self.record_alternate_paths(distribution, next_token)
            self.text_index += len(next_token)
            bits += next_bits

            if progress == "tokens":
                print(next_token, next_bits.to01())
            elif progress == "text":
                print(next_token, end="")

        return bits
