"""
Device-free checks of how a research project name read from the screen is matched to a project.

The sidecar's OCR models cannot read the D of the font the research cards use: it comes back as
0, as 1, or not at all, and a C is sometimes lost the same way. Upstream's own model made the 1
mistake, and upstream repairs it; the 0 and the missing letter are repaired here. The readings
below are the ones recorded in live runs on EN between 2026-10-18 and 2026-11-18.

Run from the repository root: python -m pytest ai_sidecar/tests/test_research_name.py
"""
import logging
from types import SimpleNamespace

import pytest

import module.research.project as project
import module.research.selector as selector
from module.research.project import ResearchProject, research_detect
from module.research.project_data import LIST_RESEARCH_PROJECT

D_AND_C = [row for row in LIST_RESEARCH_PROJECT if row['name'][0] in 'DC']


def row_id(row):
    return 'S%s-%s' % (row['series'], row['name'])


@pytest.fixture
def log():
    """Records what the research code logs, straight from ALAS's logger."""
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    project.logger.addHandler(handler)
    yield records
    project.logger.removeHandler(handler)


def lines(records, level):
    return [record.getMessage() for record in records if record.levelno == level]


# --- The readings recorded in live runs ---------------------------------------------------

LIVE_READINGS = [
    # series, as the OCR returned it, the project on the card
    (1, '0-792-MI', 'D-792-MI'),
    (1, '0-774-MI', 'D-774-MI'),
    (1, '0-318-MI', 'D-318-MI'),
    (7, '0-703-MI', 'D-703-MI'),
    (7, '0-790-M1', 'D-790-MI'),
    (8, '0-736-MI', 'D-736-MI'),
    (8, '0-764-MI', 'D-764-MI'),
    (9, '0-732-MI', 'D-732-MI'),
    (1, '-744-MI', 'D-744-MI'),
    (1, '-344-MI', 'D-344-MI'),
    (1, '-185-MI', 'C-185-MI'),
]


@pytest.mark.parametrize('series, read, name', LIVE_READINGS)
def test_a_reading_recorded_live_finds_its_project(series, read, name):
    found = ResearchProject(read, series)
    assert found.valid
    assert found.name == name
    assert found.data['name'] == name
    assert found.genre == name[0]


# --- The whole project table ---------------------------------------------------------------

@pytest.mark.parametrize('row', LIST_RESEARCH_PROJECT, ids=row_id)
def test_a_name_read_correctly_still_finds_itself(row):
    found = ResearchProject(row['name'], row['series'])
    assert found.valid
    assert found.data is row


@pytest.mark.parametrize('unread', ['0', '1', ''], ids=['letter read as 0', 'letter read as 1', 'letter missing'])
@pytest.mark.parametrize('row', D_AND_C, ids=row_id)
def test_a_d_or_c_project_is_found_without_its_letter(row, unread):
    found = ResearchProject(unread + row['name'][1:], row['series'])
    assert found.valid
    assert found.data is row
    assert found.name == row['name']


def test_upstream_guesses_for_a_missing_letter_are_kept():
    # Upstream's model loses Q, G and E. A D project with the same number but another ending
    # must not take these: S1 has D-031-UL, and S9 has no D-389 at all.
    assert ResearchProject('-031-MI', 1).data['name'] == 'E-031-MI'
    assert ResearchProject('-389-MI', 9).data['name'] == 'Q-389-MI'
    assert ResearchProject('-412-MI', 1).data['name'] == 'G-412-MI'


@pytest.mark.parametrize('row', LIST_RESEARCH_PROJECT, ids=row_id)
def test_a_missing_letter_is_the_first_of_d_c_q_g_e_that_the_series_has(row):
    # The rule in full, for every project: the name without its letter is given D, C, Q, G
    # and E in that order, and the first that names a project of the series is taken.
    series_names = {other['name'] for other in LIST_RESEARCH_PROJECT if other['series'] == row['series']}
    expected = next((genre + row['name'][1:] for genre in 'DCQGE'
                     if genre + row['name'][1:] in series_names), None)
    found = ResearchProject(row['name'][1:], row['series'])
    assert (found.data['name'] if found.valid else None) == expected


def twins():
    """(series, D or C name, the other name) where only the genre letter tells two projects apart."""
    found = []
    for row in D_AND_C:
        for other in LIST_RESEARCH_PROJECT:
            if other['series'] == row['series'] and other['name'][1:] == row['name'][1:] \
                    and other['name'] != row['name']:
                found.append((row['series'], row['name'], other['name']))
    return sorted(found)


def test_the_projects_that_differ_only_by_letter_are_the_known_five():
    # If the project table gains another pair, the warning below is what will report it live.
    assert twins() == [
        (4, 'D-063-UL', 'H-063-UL'),
        (4, 'D-068-UL', 'Q-068-UL'),
        (4, 'D-368-MI', 'Q-368-MI'),
        (6, 'D-339-MI', 'H-339-MI'),
        (8, 'D-387-MI', 'H-387-MI'),
    ]


@pytest.mark.parametrize('series, name, other', twins())
def test_a_missing_letter_is_taken_as_d_and_the_other_project_is_named_in_a_warning(log, series, name, other):
    # The sidecar reads Q and H and cannot read D, so D is the likelier card.
    found = ResearchProject(name[1:], series)
    assert found.data['name'] == name
    warnings = lines(log, logging.WARNING)
    assert len(warnings) == 1
    assert name in warnings[0] and other in warnings[0] and repr(name[1:]) in warnings[0]


# --- What the log says ---------------------------------------------------------------------

def test_a_name_read_correctly_logs_nothing(log):
    ResearchProject('G-412-MI', 1)
    assert log == []


def test_a_letter_read_as_0_is_logged_as_a_revision(log):
    ResearchProject('0-792-MI', 1)
    assert lines(log, logging.INFO) == ['Research name 0-792-MI is revised to D-792-MI']
    assert lines(log, logging.WARNING) == []


def test_a_missing_letter_is_logged_with_the_project_it_was_taken_as(log):
    ResearchProject('-744-MI', 1)
    assert "Research name '-744-MI' has no genre letter, taken as D-744-MI" in lines(log, logging.INFO)
    assert lines(log, logging.WARNING) == []


def test_a_name_matched_without_its_ending_is_logged(log):
    # Upstream matches on genre and number when the ending is wrong, and said nothing about it.
    found = ResearchProject('0-774-M', 1)
    assert found.valid and found.data['name'] == 'D-774-MI'
    assert "Research name '0-774-M' matched D-774-MI by genre and number, its ending was not read" \
        in lines(log, logging.INFO)


def test_a_c_read_as_d_is_logged(log):
    # Upstream's rule for a shining card: D-153-MI is not a project, C-153-MI is.
    found = ResearchProject('D-153-MI', 1)
    assert found.valid and found.data['name'] == 'C-153-MI'
    assert "Research name 'D-153-MI' taken as C-153-MI, the C was read as D" in lines(log, logging.INFO)


def test_a_name_with_no_letter_and_a_cut_ending_stays_invalid_and_says_what_was_read(log):
    # Recorded live on 2026-10-18. Two faults in one name; left invalid, as upstream leaves it.
    found = ResearchProject('-774-M', 1)
    assert not found.valid
    assert lines(log, logging.WARNING) == [
        "Invalid research S1 774-M: read as '-774-M', no series 1 project has that name"]


def test_an_unread_series_is_named_as_the_cause(log):
    # Recorded live on 2026-11-18: the first screenshot after the page opened gave series 0.
    found = ResearchProject('0-792-MI', 0)
    assert not found.valid
    assert lines(log, logging.WARNING) == [
        "Invalid research S0 D-792-MI: read as '0-792-MI', the series was not read"]


def test_an_empty_slot_is_named_as_such(log):
    # A saved frame of 2026-11-17 with one project started: nothing where the third card was.
    found = ResearchProject('', 0)
    assert not found.valid
    assert lines(log, logging.WARNING) == ["Invalid research S0 : read as '', no name was read"]


def test_an_unread_letter_with_a_number_that_is_neither_d_nor_c_stays_invalid(log):
    found = ResearchProject('0-389-MI', 9)
    assert not found.valid
    assert lines(log, logging.WARNING) == [
        "Invalid research S9 -389-MI: read as '0-389-MI', no series 9 project has that name"]


# --- A whole screen ------------------------------------------------------------------------

SCREENS = [
    # The names the sidecar returned for the five cards of a real screen, and the series ALAS
    # read: two saved frames read again offline, and one screen from the log of 2026-11-18.
    (['1-374-MI', 'E-118-MI', 'G-412-MI', 'H-387-MI', 'B-636-MI'], [1, 1, 7, 6, 1],
     ['D-374-MI', 'E-118-MI', 'G-412-MI', 'H-387-MI', 'B-636-MI']),
    (['Q-089-MI', 'G-531-MI', 'Q-368-MI', 'B-397-RF', '0-764-MI'], [1, 1, 7, 1, 8],
     ['Q-089-MI', 'G-531-MI', 'Q-368-MI', 'B-397-RF', 'D-764-MI']),
    (['0-792-MI', 'Q-389-MI', 'E-192-MI', '0-790-M1', '-744-MI'], [1, 9, 1, 7, 1],
     ['D-792-MI', 'Q-389-MI', 'E-192-MI', 'D-790-MI', 'D-744-MI']),
]


@pytest.mark.parametrize('read, series, names', SCREENS)
def test_every_card_of_a_real_screen_is_valid(monkeypatch, read, series, names):
    monkeypatch.setattr(project, 'get_research_name', lambda image: read)
    monkeypatch.setattr(project, 'get_research_series_3', lambda image: series)
    found = research_detect(image=None)
    assert [p.data['name'] if p.valid else None for p in found] == names
    assert [p.raw_name for p in found] == read


class Clock:
    """Stands in for the selector's Timer: reached on the given check."""

    def __init__(self, reached_on):
        self.checks, self.reached_on = 0, reached_on

    def start(self):
        return self

    def reached(self):
        self.checks += 1
        return self.checks >= self.reached_on


def run_selector(monkeypatch, screens, reached_on):
    """Run ResearchSelector.research_detect over a list of screens, one per screenshot."""
    shots = iter(screens)
    monkeypatch.setattr(selector, 'research_detect',
                        lambda image: [ResearchProject(name, series) for name, series in next(shots)])
    monkeypatch.setattr(selector, 'Timer', lambda *args, **kwargs: Clock(reached_on))
    chosen = object.__new__(selector.ResearchSelector)
    chosen.config = SimpleNamespace(SERVER='en')
    chosen.device = SimpleNamespace(image=None, sleep=lambda seconds: None, screenshot=lambda: None)
    chosen.research_detect()
    return chosen.projects


GOOD = [('G-412-MI', 1), ('E-118-MI', 1), ('Q-368-MI', 7), ('B-397-RF', 1), ('G-531-MI', 1)]
BAD = [('-774-M', 1), ('E-118-MI', 1), ('0-792-MI', 0), ('B-397-RF', 1), ('G-531-MI', 1)]


def test_when_the_trials_run_out_the_selector_names_what_is_still_invalid(monkeypatch, log):
    projects = run_selector(monkeypatch, [BAD, BAD], reached_on=2)
    assert [p.valid for p in projects] == [False, True, False, True, True]
    warnings = lines(log, logging.WARNING)
    assert warnings[-2] == 'Failed to OCR research name after 3 trial, assume correct'
    assert warnings[-1] == ("Research names still invalid: S1 774-M (Invalid) read as '-774-M', "
                            "S0 D-792-MI (Invalid) read as '0-792-MI'")


def test_a_screen_that_reads_cleanly_gets_no_such_line(monkeypatch, log):
    projects = run_selector(monkeypatch, [GOOD], reached_on=5)
    assert all(p.valid for p in projects)
    assert lines(log, logging.WARNING) == []
