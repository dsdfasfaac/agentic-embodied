"""Batching preserves durable action boundaries and atomic critic evidence."""
import json
import sqlite3

import pytest

from robots.arx.gateway.journal import Journal


def test_post_action_batch_visible_atomically_after_durable_step_commit(tmp_path):
    path = tmp_path / 'journal.sqlite3'
    journal = Journal(path)
    reader = sqlite3.connect(path)
    journal.record('step_intent', {'step': 1})
    journal.record('step_commit', {'step': 1})
    assert reader.execute('SELECT count(*) FROM records').fetchone()[0] == 2
    with journal.batch():
        journal.record('real_feature_evidence', {'step': 1})
        journal.record('critic_assessment', {'status': 'failure'})
        journal.save_snapshot({'state': 'INTERRUPTED'})
        assert reader.execute('SELECT count(*) FROM records').fetchone()[0] == 2
    assert reader.execute('SELECT count(*) FROM records').fetchone()[0] == 4
    assert json.loads(reader.execute('SELECT payload FROM snapshot').fetchone()[0])['state'] == 'INTERRUPTED'
    assert journal.db.execute('PRAGMA synchronous').fetchone()[0] == 2
    reader.close()
    journal.db.close()


def test_failed_post_action_batch_keeps_physical_truth_and_rolls_back_partial_assessment(tmp_path):
    journal = Journal(tmp_path / 'journal.sqlite3')
    journal.record('step_commit', {'step': 1})
    with pytest.raises(RuntimeError):
        with journal.batch():
            journal.record('real_feature_evidence', {'step': 1})
            raise RuntimeError('critic failed')
    assert journal.db.execute('SELECT kind FROM records').fetchall() == [('step_commit',)]
    journal.record('execution_error', {'after_step': 1})
    assert journal.db.execute('SELECT count(*) FROM records').fetchone()[0] == 2
    journal.db.close()
