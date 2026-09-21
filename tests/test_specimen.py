"""Unit tests for THE SPECIMEN living organism backend and FastAPI endpoints."""

import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from specimen.organism import SpecimenOrganism
from specimen.server import app, organism


@pytest.fixture
def test_organism(tmp_path):
    db_file = tmp_path / "test_specimen.db"
    return SpecimenOrganism(db_path=str(db_file))


@pytest.fixture
def client():
    return TestClient(app)


class TestSpecimenOrganism:
    def test_initial_state(self, test_organism):
        state = test_organism.get_state()
        assert "SPECIMEN #001" in state["specimen_id"]
        assert state["total_feeds"] == 0
        assert state["sleep_count"] == 0
        assert not state["is_sleeping"]
        assert len(state["recent_dreams"]) == 8

    def test_mood_mapping(self, test_organism):
        assert test_organism._get_mood(0.05) == "calm"
        assert test_organism._get_mood(0.18) == "uneasy"
        assert test_organism._get_mood(0.35) == "alarmed"
        assert test_organism._get_mood(0.75) == "in pain"

    def test_feed_clean_input(self, test_organism):
        clean_img = np.zeros((28, 28), dtype=np.float32)
        clean_img[10:18, 14] = 1.0  # Vertical line (1)
        res = test_organism.feed(clean_img)

        assert "prediction" in res
        assert 0 <= res["prediction"] <= 9
        assert 0.0 <= res["confidence"] <= 1.0
        assert 0.0 <= res["p_error"] <= 1.0
        assert res["mood"] in ["calm", "uneasy", "alarmed", "in pain"]
        assert not np.isnan(res["p_error"])
        assert len(test_organism.memory_buffer) == 1

    def test_reveal_flow(self, test_organism):
        img = np.random.rand(28, 28).astype(np.float32)
        feed_res = test_organism.feed(img)
        pred = feed_res["prediction"]

        # Reveal correct label
        rev_correct = test_organism.reveal(pred)
        assert rev_correct["was_correct"] is True

        # Feed another and reveal wrong label
        img2 = np.random.rand(28, 28).astype(np.float32)
        feed_res2 = test_organism.feed(img2)
        wrong_label = (feed_res2["prediction"] + 1) % 10
        rev_wrong = test_organism.reveal(wrong_label)
        assert rev_wrong["was_correct"] is False
        assert rev_wrong["true_label"] == wrong_label

    def test_sleep_cycle(self, test_organism):
        # Feed some samples
        for _ in range(5):
            test_organism.feed(np.random.rand(28, 28).astype(np.float32))

        sleep_res = test_organism.sleep_cycle()
        assert sleep_res["status"] == "success"
        assert sleep_res["sleep_count"] == 1
        assert len(test_organism.recent_dreams) > 0


class TestSpecimenAPI:
    def test_get_state(self, client):
        res = client.get("/state")
        assert res.status_code == 200
        data = res.json()
        assert "SPECIMEN #001" in data["specimen_id"]
        assert "vitals" in data

    def test_get_history(self, client):
        res = client.get("/history")
        assert res.status_code == 200
        data = res.json()
        assert "memory_buffer" in data
        assert "event_log" in data

    def test_post_feed_and_reveal(self, client):
        sample_img = [0.0] * 784
        sample_img[350] = 1.0
        feed_res = client.post("/feed", json={"image": sample_img})
        assert feed_res.status_code == 200
        data = feed_res.json()
        assert "prediction" in data

        # Reveal
        rev_res = client.post("/reveal", json={"label": data["prediction"]})
        assert rev_res.status_code == 200
        assert rev_res.json()["was_correct"] is True

    def test_get_preset(self, client):
        res = client.get("/preset/clean_3")
        assert res.status_code == 200
        assert len(res.json()["image"]) == 784
