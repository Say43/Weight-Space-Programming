"""Finds the letter tokens of the answer options."""
from eval.prompts import LETTERS


def option_positions(tok, ids):
    """Position of the letter token of each option; returns a dict letter -> position."""
    found = {}
    pieces = [tok.decode([t]) for t in ids]
    for pos, piece in enumerate(pieces):
        stripped = piece.strip()
        if len(stripped) == 2 and stripped[0] == "(" and stripped[1] in LETTERS and pieces[pos + 1].startswith(")"):
            found.setdefault(stripped[1], pos)
        elif stripped in LETTERS and pieces[pos - 1].strip() == "(" and pieces[pos + 1].startswith(")"):
            found.setdefault(stripped, pos)
    return found
