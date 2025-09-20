import unittest
import translator
import random
from string import ascii_lowercase
import itertools
from bitarray import bitarray
from bitarray.util import random_p
from typing import Optional

repeats = 1000

tp = translator.TokenProbability
ap = translator.AlternatePath


class TestPredictionModel(translator.PredictionModel):
    def __init__(self, model: dict[str, list[tuple[str, float]]]):
        self.model = model

    def _predict_next(self, text: str) -> list[tp]:
        return sorted(
            [tp(*token) for token in self.model[text]],
            key=lambda token: token.prob,
        )

    def _predict_start(self):
        return self._predict_next("")


class RandomPredictionModel(translator.PredictionModel):
    consonants = "bc"
    vowels = "ae"

    all_letters = consonants + vowels

    class TokenPrefixTree:
        def __init__(self, text):
            self.text = text
            self.children = None
            self.real = True
            self.included = True

        def add_children(
            self, children: list["RandomPredictionModel.TokenPrefixTree"], real: bool
        ):
            assert len(children) >= 2
            assert all(e.text[:-1] == self.text for e in children)
            assert set(e.text[-1] for e in children) in [
                set(RandomPredictionModel.consonants),
                set(RandomPredictionModel.vowels),
            ]
            self.children = children
            self.real = real
            if not self.real:
                self.included = False

        def real_descendents(self) -> list["RandomPredictionModel.TokenPrefixTree"]:
            descendents = []
            if self.real and len(self.text) > 1:
                descendents.append(self)
            if self.children:
                for child in self.children:
                    descendents += child.real_descendents()
            return descendents

        def __repr__(self):
            return self._repr_helper(True)

        def _repr_helper(self, full: bool) -> str:
            if full:
                start = self.text
            else:
                start = self.text[-1]
            if not self.real:
                start = start + "!"

            if self.children is None:
                return start
            return (
                start
                + "("
                + "".join(child._repr_helper(False) for child in self.children)
                + ")"
            )

    def _filter_complete_tokens(self, tokens: list[str]) -> list[str]:
        max_len = max(len(token) for token in tokens)
        tokens_by_length: list[list["RandomPredictionModel.TokenPrefixTree"]] = [
            [] for _ in range(max_len + 1)
        ]
        token_trees = [self.TokenPrefixTree(token) for token in tokens]
        for token in token_trees:
            tokens_by_length[len(token.text)].append(token)

        for tokens_of_length_n, tokens_of_length_nminus1 in itertools.pairwise(
            reversed(tokens_by_length)
        ):
            checked = set()
            for token in tokens_of_length_n:
                if token in checked:
                    continue

                token_prefix = token.text[:-1]
                token_end = token.text[-1]

                if token_end in self.consonants:
                    all_end_chars = self.consonants
                else:
                    all_end_chars = self.vowels

                complement_candidates = [
                    token
                    for token in tokens_of_length_n
                    if token.text[:-1] == token_prefix
                ]
                checked &= set(complement_candidates)

                complement_ends = [
                    complement_candidate.text[-1]
                    for complement_candidate in complement_candidates
                ]

                if set(all_end_chars) == set(complement_ends):
                    prefix_token: Optional[RandomPredictionModel.TokenPrefixTree] = None
                    for shorter_token in tokens_of_length_nminus1:
                        if shorter_token.text == token_prefix:
                            prefix_token = shorter_token

                    real = not (prefix_token is None or not prefix_token.real)
                    if prefix_token is None:
                        prefix_token = self.TokenPrefixTree(token_prefix)
                        tokens_of_length_nminus1.append(prefix_token)

                    prefix_token.add_children(complement_candidates, real)

        for token in token_trees:
            if token.children:
                random.choice(token.real_descendents()).included = False

        return [token.text for token in token_trees if token.included]

    def _predict_start(self) -> list[tp]:
        return sorted(
            [tp(letter, 1 / len(self.all_letters)) for letter in self.all_letters],
            key=lambda token: token[-1::-1],
        )

    def _predict_next(self, text: str) -> list[tp]:
        random.seed(text)

        if text[-1] in self.consonants:
            next_pattern = [self.vowels, self.consonants]
        else:
            next_pattern = [self.consonants, self.vowels]

        tokens = list(next_pattern[0])
        tokens_to_generate = random.randint(10, 100)
        tokens_generated = 0
        while tokens_generated < tokens_to_generate:
            token = ""
            pattern = itertools.cycle(next_pattern)
            while 0 <= len(token) < 10:
                token += random.choice(next(pattern))
                if random.random() < 0.3:
                    break
            if token not in tokens:
                tokens.append(token)
                tokens_generated += 1

        tokens = self._filter_complete_tokens(tokens)

        power = random.uniform(0.1, 3)
        probs = [
            random.uniform(0.5, 1.5) / (i**power) for i in range(1, len(tokens) + 1)
        ]
        total_prob = sum(probs)
        normalized_probs = [prob / total_prob for prob in probs]
        random.shuffle(normalized_probs)

        random.seed()

        return sorted(
            [tp(token, prob) for token, prob in zip(tokens, normalized_probs)],
            key=lambda token: token[-1::-1],
        )

    def random_starter_text(self, length) -> str:
        next_pattern = [self.consonants, self.vowels]
        pattern = itertools.cycle(next_pattern)
        if random.random() < 0.5:
            next(pattern)

        return "".join(random.choice(next(pattern)) for _ in range(length))


class ETETestCase(unittest.TestCase):
    def assertEqualDists(self, first: list[tp], second: list[tp], delta=None):
        first_sorted = sorted(first, key=lambda tp: tp.text)
        second_sorted = sorted(second, key=lambda tp: tp.text)

        self.assertEqual(
            [tp.text for tp in first_sorted],
            [tp.text for tp in second_sorted],
            msg=f"\nDiffering tokens:\n{first_sorted}\n{second_sorted}",
        )

        for tp1, tp2 in zip(first_sorted, second_sorted):
            self.assertAlmostEqual(
                tp1.prob,
                tp2.prob,
                delta=delta,
                msg=f"\nDiffering probabilities:\n{first_sorted}\n{second_sorted}",
            )

    def assertEqualAlternates(
        self,
        first: list[ap],
        second: list[ap],
    ):
        self.assertEqual(sorted(first), sorted(second))


class TestHuffmanCode(ETETestCase):
    def _sample_distribution(self, n: int) -> list[tp]:
        probs = [(2 ** random.uniform(-1, 1)) / i for i in range(1, n + 1)]
        total_prob = sum(probs)
        token_iter = itertools.product(ascii_lowercase, repeat=4)
        return [tp("".join(next(token_iter)), prob / total_prob) for prob in probs]

    def test_simple_case(self):
        """Simple example of a Huffman code.

        Taken from the Wikipedia article. The specific encodings might be different but the lengths should be the same; that is all this checks for.
        """

        data = [
            (" ", 7, "111"),
            ("a", 4, "010"),
            ("e", 4, "000"),
            ("f", 3, "1101"),
            ("h", 2, "1010"),
            ("i", 2, "1000"),
            ("m", 2, "0111"),
            ("n", 2, "0010"),
            ("s", 2, "1011"),
            ("t", 2, "0110"),
            ("l", 1, "11001"),
            ("o", 1, "00110"),
            ("p", 1, "10011"),
            ("r", 1, "11000"),
            ("u", 1, "00111"),
            ("x", 1, "10010"),
        ]

        distribution = [tp(text, prob) for text, prob, _ in data]
        expected_codes = {text: bitarray(code) for text, _, code in data}

        hc = translator.generate_huffman_code(distribution)
        for text, expected_code in expected_codes.items():
            generated_code = hc.token_to_bits(text)
            self.assertIsNotNone(generated_code)
            assert generated_code is not None
            self.assertEqual(len(generated_code), len(expected_code))

    def test_order_doesnt_matter(self):
        """If all the probabilities are different, the order does not affect the resulting code."""
        for _ in range(repeats):
            distribution = self._sample_distribution(100)
            shuffled_dist = distribution[:]
            random.shuffle(shuffled_dist)
            hc1 = translator.generate_huffman_code(distribution)
            hc2 = translator.generate_huffman_code(shuffled_dist)
            self.assertTrue(
                all(
                    hc1.token_to_bits(token.text) == hc2.token_to_bits(token.text)
                    for token in distribution
                )
            )

    def test_zero_default(self):
        """When it reaches the end of the input bits but needs more bits, acts as if there are extra zeros at end"""

        data = [("a", 0.4), ("b", 0.25), ("c", 0.35)]
        distribution = [tp(text, prob) for text, prob in data]
        hc = translator.generate_huffman_code(distribution)
        self.assertEqual(hc.token_to_bits("a"), bitarray("0"))
        self.assertEqual(hc.token_to_bits("b"), bitarray("10"))
        self.assertEqual(hc.token_to_bits("c"), bitarray("11"))

        self.assertEqual(hc.bits_to_token(bitarray("1"), 0), ("b", 2))


class TestExpansion(ETETestCase):
    def test_normal_case(self):
        """If all tokens have probability < threshold, returns the distribution unchanged"""

        pm = TestPredictionModel(
            {
                "hello": [
                    ("!", 0.3),
                    (" world", 0.25),
                    (" there", 0.2),
                    (" kitty", 0.15),
                    (" darkness", 0.1),
                ]
            }
        )
        dist = pm.predict("hello")

        ttb = translator.TextToBits(pm, "hello_ ")
        ttb.text_index = 5
        ttb_dist = ttb.predict_tokens()

        self.assertEqual(dist, ttb_dist)

    def test_expansion_threshold(self):
        """If a token has prob > threshold, expands into ngrams"""

        pm = TestPredictionModel(
            {
                "a": [
                    ("b", 0.6),
                    ("c", 0.4),
                ],
                "ab": [
                    ("w", 0.1),
                    ("x", 0.2),
                    ("y", 0.3),
                    ("z", 0.4),
                ],
            }
        )

        expected_dist = [
            tp("b", 0.06),
            tp("bx", 0.12),
            tp("by", 0.18),
            tp("bz", 0.24),
            tp("c", 0.4),
        ]

        ttb = translator.TextToBits(pm, "a ")
        ttb.text_index = 1
        ttb_dist = ttb.predict_tokens()

        self.assertEqualDists(expected_dist, ttb_dist)

    def test_expansion_tolerance(self):
        """Different expansion tolerance leads to different result"""

        pm = TestPredictionModel(
            {
                "a": [
                    ("b", 0.6),
                    ("c", 0.4),
                ],
                "ab": [
                    ("w", 0.1),
                    ("x", 0.2),
                    ("y", 0.3),
                    ("z", 0.4),
                ],
            }
        )

        expected_dist = [
            tp("b", 0.18),
            tp("by", 0.18),
            tp("bz", 0.24),
            tp("c", 0.4),
        ]

        ttb = translator.TextToBits(pm, "a ", expansion_tolerance=0.2)
        ttb.text_index = 1
        ttb_dist = ttb.predict_tokens()

        self.assertEqualDists(expected_dist, ttb_dist)

    def test_expand_multiple(self):
        """Lower threshold, several tokens greater than threshold"""

        pm = TestPredictionModel(
            {
                "a": [
                    ("b", 0.4),
                    ("c", 0.4),
                    ("d", 0.2),
                ],
                "ab": [
                    ("x", 0.4),
                    ("y", 0.4),
                    ("z", 0.2),
                ],
                "ac": [
                    ("x", 0.4),
                    ("y", 0.4),
                    ("z", 0.2),
                ],
            }
        )

        expected_dist = [
            tp("bx", 0.16),
            tp("by", 0.16),
            tp("b", 0.08),
            tp("cx", 0.16),
            tp("cy", 0.16),
            tp("c", 0.08),
            tp("d", 0.2),
        ]

        ttb = translator.TextToBits(pm, "a ", expansion_threshold=0.3)
        ttb.text_index = 1
        ttb_dist = ttb.predict_tokens()

        self.assertEqualDists(expected_dist, ttb_dist)

    def test_expand_deep(self):
        """A 2-gram is over the threshold; need to expand into 3-grams"""

        pm = TestPredictionModel(
            {
                "a": [
                    ("b", 0.8),
                    ("x", 0.2),
                ],
                "ab": [
                    ("c", 0.8),
                    ("y", 0.2),
                ],
                "abc": [
                    ("m", 0.4),
                    ("n", 0.3),
                    ("o", 0.3),
                ],
            }
        )

        expected_dist = [
            tp("bcm", 0.256),
            tp("bcn", 0.192),
            tp("bco", 0.192),
            tp("bc", 0),
            tp("by", 0.16),
            tp("b", 0),
            tp("x", 0.2),
        ]

        ttb = translator.TextToBits(pm, "a ")
        ttb.text_index = 1
        ttb_dist = ttb.predict_tokens()

        self.assertEqualDists(expected_dist, ttb_dist)


class TestFiltering(ETETestCase):
    def test_simple_filter_case_btt(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bc", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("d", 0.25),
                    ("e", 0.35),
                ],
            }
        )

        btt = translator.BitsToText(pm, bitarray("000"), expansion_threshold=1)
        self.assertEqual("abd", btt.translate())

    def test_simple_filter_non_case_btt(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bx", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("d", 0.25),
                    ("e", 0.35),
                ],
            }
        )

        btt = translator.BitsToText(pm, bitarray("000"), expansion_threshold=1)
        self.assertEqual("abc", btt.translate())

    def test_simple_filter_case_ttb(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bc", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("d", 0.25),
                    ("e", 0.35),
                ],
            }
        )

        ttb = translator.TextToBits(pm, "ab", expansion_threshold=1)
        ttb.translate()
        self.assertEqualDists(
            [tp("d", 0.25), tp("e", 0.35)], ttb.generate_distribution()
        )

    def test_simple_filter_non_case_ttb(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bc", 0.6),
                ],
                "ab": [
                    ("x", 0.4),
                    ("d", 0.25),
                    ("e", 0.35),
                ],
            }
        )

        ttb = translator.TextToBits(pm, "ab", expansion_threshold=1)
        ttb.translate()
        self.assertEqualDists(
            [tp("x", 0.4), tp("d", 0.25), tp("e", 0.35)], ttb.generate_distribution()
        )

    def test_deep_filter_case_btt(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bcd", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("d", 0.6),
                ],
                "abc": [
                    ("d", 0.4),
                    ("e", 0.25),
                    ("f", 0.35),
                ],
            }
        )

        btt = translator.BitsToText(pm, bitarray("0000"), expansion_threshold=1)
        self.assertEqual("abce", btt.translate())

    def test_deep_filter_non_case_btt(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("z", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bcx", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("d", 0.6),
                ],
                "abc": [
                    ("d", 0.4),
                    ("e", 0.25),
                    ("f", 0.35),
                ],
            }
        )

        btt = translator.BitsToText(pm, bitarray("0000"), expansion_threshold=1)
        self.assertEqual("abcd", btt.translate())

    def test_alternate_paths_are_filtered(self):
        pm = TestPredictionModel(
            {
                "": [
                    ("a", 0.4),
                    ("w", 0.6),
                ],
                "a": [
                    ("b", 0.4),
                    ("bcd", 0.6),
                ],
                "ab": [
                    ("c", 0.4),
                    ("x", 0.6),
                ],
                "abc": [
                    ("d", 0.4),
                    ("y", 0.25),
                    ("z", 0.35),
                ],
                "abx": [
                    ("m", 0.4),
                    ("n", 0.6),
                ],
                "abcy": [
                    ("a", 0.4),
                    ("b", 0.6),
                ],
            }
        )

        btt = translator.BitsToText(pm, bitarray("0"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(0, "w")])

        btt = translator.BitsToText(pm, bitarray("00"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(1, "bcd")])

        btt = translator.BitsToText(pm, bitarray("000"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(1, "bcd"), ap(2, "x")])

        btt = translator.BitsToText(pm, bitarray("0010"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(3, "n")])

        btt = translator.BitsToText(pm, bitarray("0000"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(1, "bcd"), ap(3, "z")])

        btt = translator.BitsToText(pm, bitarray("00000"), expansion_threshold=1)
        btt.translate()
        self.assertEqualAlternates(btt.alternate_paths, [ap(4, "b")])


class TestReversibility(ETETestCase):
    def test_ttbtt(self):
        tries = 0
        while tries < repeats:
            try:
                rpm = RandomPredictionModel()
                in_text = rpm.random_starter_text(100)
                ttb = translator.TextToBits(rpm, in_text)
                bits = ttb.translate()
                btt = translator.BitsToText(rpm, bits)
                out_text = btt.translate()
                self.assertEqual(in_text, out_text)

                tries += 1
            except translator.WontHappenInPracticeException:
                continue
            except:
                raise

    def test_btttb(self):
        tries = 0
        while tries < repeats:
            try:
                rpm = RandomPredictionModel()
                in_bits = random_p(100)
                btt = translator.BitsToText(rpm, in_bits)
                text = btt.translate()
                ttb = translator.TextToBits(rpm, text)
                out_bits = ttb.translate()

                self.assertEqual(in_bits, out_bits[: len(in_bits)])
                self.assertEqual(
                    bitarray(len(out_bits) - len(in_bits)), out_bits[len(in_bits) :]
                )

                tries += 1
            except translator.WontHappenInPracticeException:
                continue
            except:
                raise


# if __name__ == "__main__":
#     unittest.main()


rpm = RandomPredictionModel()
in_bits = random_p(100)
btt = translator.BitsToText(rpm, in_bits)
btt.translate()
print(btt.timers)
