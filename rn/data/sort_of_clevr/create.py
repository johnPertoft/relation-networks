import itertools
import random
from pathlib import Path
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from PIL import ImageDraw


IMG_SIZE = 75
OBJECT_SIZE = 10

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta"
]

RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255)
}

SHAPES = [
    "square",
    "circle"
]


def _draw_square_fn(o, draw):
    x, y = o.pos
    r = OBJECT_SIZE // 2
    draw.rectangle([(x - r, y - r), (x + r, y + r)], RGB[o.color])


def _draw_circle_fn(o, draw):
    x, y = o.pos
    r = OBJECT_SIZE // 2
    draw.ellipse([(x - r, y - r), (x + r, y + r)], RGB[o.color])


SHAPE_DRAW_FNS = {
    "square": _draw_square_fn,
    "circle": _draw_circle_fn
}


DIRECTIONS = [
    "left",
    "right",
    "top",
    "bottom"
]


class Object:
    def __init__(self, color: str, shape: str, position: Tuple[int, int]):
        self.color = color
        self.shape = shape
        self.pos = position


class Answer:
    """Represents a possible answer for the questions in this Sort-of-CLEVR implementation."""

    dim = len(COLORS) + len(SHAPES) + len(DIRECTIONS) + 1

    def __init__(self,
                 color: Optional[str]=None,
                 shape: Optional[str]=None,
                 direction: Optional[str]=None,
                 count: Optional[int]=None):

        assert sum(a is not None for a in [color, shape, direction, count]) == 1, "Exactly one answer should be set."
        self._color = color
        self._shape = shape
        self._direction = direction
        self._count = count

    def __repr__(self):
        if self._color is not None:
            return self._color
        elif self._shape is not None:
            return self._shape
        elif self._direction is not None:
            return self._direction
        else:
            return f"{self._count}"

    def encode(self):
        color_answer = np.zeros((len(COLORS),))
        if self._color is not None:
            color_answer[COLORS.index(self._color)] = 1

        shape_answer = np.zeros((len(SHAPES),))
        if self._shape is not None:
            shape_answer[SHAPES.index(self._shape)] = 1

        directions_answer = np.zeros((len(DIRECTIONS),))
        if self._direction is not None:
            directions_answer[DIRECTIONS.index(self._direction)] = 1

        count_answer = np.zeros((1,))
        if self._count is not None:
            count_answer[0] = self._count

        return np.hstack((color_answer, shape_answer, directions_answer, count_answer))

    @classmethod
    def decode(cls, answer: np.ndarray) -> "Answer":
        color_dim_start = 0
        shape_dim_start = len(COLORS)
        direction_dim_start = shape_dim_start + len(SHAPES)
        count_dim = direction_dim_start + len(DIRECTIONS)

        color_answer = answer[color_dim_start:shape_dim_start]
        color_idx = np.where(color_answer == 1)[0]
        if color_idx.size > 0:
            return cls(color=COLORS[color_idx[0]])

        shape_answer = answer[shape_dim_start:direction_dim_start]
        shape_idx = np.where(shape_answer == 1)[0]
        if shape_idx.size > 0:
            return cls(shape=SHAPES[shape_idx[0]])

        direction_answer = answer[direction_dim_start:count_dim]
        direction_idx = np.where(direction_answer == 1)[0]
        if direction_idx.size > 0:
            return cls(direction=DIRECTIONS[direction_idx[0]])

        return cls(count=answer[count_dim])


class Question:
    """Pairs a question with a function to compute the answers to it given the objects in the scene."""

    def __init__(self, question_fmt: str, relational: bool, answer_fn: Callable[[List[Object], str], Answer]):
        self._question_fmt = question_fmt
        self._relational = relational
        self._answer_fn = answer_fn

    def question(self, color: str) -> str:
        return self._question_fmt.format(color=color)

    @property
    def relational(self) -> bool:
        return self._relational

    def answer(self, objects: List[Object], color: str) -> Answer:
        return self._answer_fn(objects, color)


def _find_object(objects: List[Object], color: str) -> Object:
    for o in objects:
        if o.color == color:
            return o


def _distance_to(o: Object) -> Callable[[Object], float]:
    def dist(p):
        return (o.pos[0] - p.pos[0]) ** 2 + (o.pos[1] - p.pos[1]) ** 2
    return dist


def _answer_shape_of_object(objects: List[Object], color: str) -> Answer:
    return Answer(shape=_find_object(objects, color).shape)


def _answer_left_or_right(objects: List[Object], color: str) -> Answer:
    return Answer(direction="left" if _find_object(objects, color).pos[0] < IMG_SIZE // 2 else "right")


def _answer_top_or_bottom(objects: List[Object], color: str) -> Answer:
    return Answer(direction="top" if _find_object(objects, color).pos[1] < IMG_SIZE // 2 else "bottom")


def _answer_shape_of_closest(objects: List[Object], color: str) -> Answer:
    o = _find_object(objects, color)
    others = [p for p in objects if p is not o]
    p = min(others, key=_distance_to(o))
    return Answer(shape=p.shape)


def _answer_shape_of_furthest(objects: List[Object], color: str) -> Answer:
    o = _find_object(objects, color)
    others = [p for p in objects if p is not o]
    p = max(others, key=_distance_to(o))
    return Answer(shape=p.shape)


def _answer_how_many_of_shape(objects: List[Object], color: str) -> Answer:
    o = _find_object(objects, color)
    return Answer(count=len([p for p in objects if p.shape == o.shape]))


# TODO: Add questions where answer is a color.


class Questions:
    """Defines an ordering of some relational and non-relational questions."""

    relational_questions = [
        Question("What is the shape of the object that is closest to the {color} object?", True, _answer_shape_of_closest),
        Question("What is the shape of the object that is furthest from the {color} object?", True, _answer_shape_of_furthest),
        Question("How many objects have the shape of the {color} object?", True, _answer_how_many_of_shape)
    ]

    non_relational_questions = [
        Question("What is the shape of the {color} object?", False, _answer_shape_of_object),
        Question("Is the {color} object on the left or right side of the image?", False, _answer_left_or_right),
        Question("Is the {color} object on the top or bottom part of the image?", False, _answer_top_or_bottom)
    ]

    assert len(relational_questions) == len(non_relational_questions), \
        "Encoding of questions requires same number of relational and non-relational questions."
    n_questions = len(relational_questions)

    dim = 2 + n_questions + len(COLORS)

    @staticmethod
    def encode(relational_question: bool, question_idx: int, color_idx: int) -> np.ndarray:
        question_type = np.zeros((2,))
        question_type[0 if relational_question else 1] = 1

        question_subtype = np.zeros((Questions.n_questions,))
        question_subtype[question_idx] = 1

        question_color = np.zeros((len(COLORS),))
        question_color[color_idx] = 1

        return np.hstack((question_type, question_subtype, question_color))

    @staticmethod
    def decode(question: np.ndarray) -> Question:
        # TODO: needed for use when reading just encoded questions.
        pass

    @staticmethod
    def generate(objects: List[Object]) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
        def sample_questions(questions, n) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
            possible_questions = list(itertools.product(range(Questions.n_questions), COLORS))
            random.shuffle(possible_questions)

            for question_idx, color in possible_questions[:n]:
                q = questions[question_idx]
                a = q.answer(objects, color)

                q_str = q.question(color)
                a_str = str(a)

                q_enc = Questions.encode(q.relational, question_idx, COLORS.index(color))
                a_enc = a.encode()

                yield q_str, a_str, q_enc, a_enc

        yield from sample_questions(Questions.relational_questions, 10)
        yield from sample_questions(Questions.non_relational_questions, 10)


_BACKGROUND_IMG = (np.ones((IMG_SIZE, IMG_SIZE, 3)) * np.array([[[128, 128, 128]]])).astype(np.uint8)


def create_image(objects: List[Object]) -> np.ndarray:
    img = Image.fromarray(_BACKGROUND_IMG)
    draw = ImageDraw.Draw(img)

    for o in objects:
        SHAPE_DRAW_FNS[o.shape](o, draw)

    return np.array(img)


def random_objects() -> List[Object]:
    def random_pos():
        margin = OBJECT_SIZE // 2
        lower, upper = margin, IMG_SIZE - margin
        return random.randint(lower, upper), random.randint(lower, upper)

    # TODO: Make sure objects don't overlap too much.
    # TODO: Potentially make which objects are closest/furthest more visually distinguishable.

    objects = [Object(color, random.choice(SHAPES), random_pos()) for color in COLORS]

    return objects


def generator(include_human_readable=True) -> Iterator[Tuple[np.ndarray, str, str, np.ndarray, np.ndarray]]:
    """Generator for Sort-of-CLEVR data points."""

    while True:
        objects = random_objects()
        img = create_image(objects)
        for question, answer, question_enc, answer_enc in Questions.generate(objects):
            if include_human_readable:
                yield img, question_enc, answer_enc, question, answer
            else:
                yield img, question_enc, answer_enc


def tfrecords(directory: Path) -> None:
    # TODO: Only store the encoded questions and answers.
    pass
