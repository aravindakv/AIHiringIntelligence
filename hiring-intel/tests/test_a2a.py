from fastapi.testclient import TestClient
from hiring_intel import a2a


def test_agent_card_and_message_send():
    c = TestClient(a2a.server)
    card = c.get("/agents/offers/.well-known/agent-card.json").json()
    assert card["skills"][0]["id"] == "offers.analyze"
    out = a2a.call_agent("offers", {"data_path": "data/raw/baseline_small", "run_id": "a2a-test"}, http=c)
    assert out.agent == "offers" and out.insights