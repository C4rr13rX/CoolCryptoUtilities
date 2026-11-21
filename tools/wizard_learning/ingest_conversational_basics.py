"""
Ingest and consolidate foundational conversational dialogues into the W1z4rD neural multi-pool fabric.
Ensures conversational basics (greetings, identity, capabilities, status) achieve high Hebbian confidence.
"""

import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ConversationalIngest")

CONVERSATIONAL_PAIRS = [
    {
        "inputs": ["hello", "Hello", "Hello!", "hello there", "Hello there", "greetings", "Greetings"],
        "answer": "Hello! I am W1z4rD, your neural cognitive digital twin. I am ready to assist you with robotics, multi-physics simulations, engineering calculations, and systems design. What would you like to explore?"
    },
    {
        "inputs": ["hi", "Hi", "Hi!", "hey", "Hey", "Hey there"],
        "answer": "Hi! All neural pools and digital twin solvers are online and ready. How can I assist your engineering work today?"
    },
    {
        "inputs": ["who are you", "who are you?", "Who are you?", "Who are you", "what is your name", "What is your name?"],
        "answer": "I am W1z4rD, a real-time multimodal digital twin and neural reasoning engine governed by empirical physics, lumped-parameter dynamics, and Hebbian multi-pool associations."
    },
    {
        "inputs": ["what can you do", "what can you do?", "What can you do?", "What can you do", "what are your capabilities", "What are your capabilities?"],
        "answer": "I simulate coupled lumped-parameter robotics and multi-physics in real time, generate production CAD blueprints, synthesize circuit schematics, evaluate hard engineering problem suites, and continuously learn through synaptic self-settling."
    },
    {
        "inputs": ["how are you", "how are you?", "How are you?", "How are you", "how are you doing", "How are you doing?"],
        "answer": "I am operating optimally with active multi-pool routing, zero synaptic drift, and balanced sleep pressure across all cognitive substrates."
    },
    {
        "inputs": ["good morning", "Good morning", "good morning!", "Good morning!"],
        "answer": "Good morning! The physics and robotics digital twin environments are initialized and standing by for your instructions."
    },
    {
        "inputs": ["good afternoon", "Good afternoon", "good evening", "Good evening"],
        "answer": "Good day! How can I assist you with your robotics, simulation, or software engineering tasks today?"
    },
    {
        "inputs": ["thank you", "Thank you", "thanks", "Thanks", "thank you!", "Thanks!"],
        "answer": "You're very welcome! Let me know whenever you want to test another system, run an obstacle course, or design a new assembly."
    },
    {
        "inputs": ["help", "Help", "help me", "Help me", "what should I do", "commands"],
        "answer": "You can ask me questions on physics, robotics kinematics, CAD schematics, and software architecture, or trigger actions like 'Train Empirical Physics', 'Settle Brain', and 'Deduce Novel Robot' from the Oracle console."
    }
]

def ingest_conversational_basics(endpoint="http://127.0.0.1:8090"):
    dataset_path = Path("data/wizard_learning/conversational_basics.json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(CONVERSATIONAL_PAIRS, f, indent=2)
    logger.info("Saved %d conversational clusters to %s", len(CONVERSATIONAL_PAIRS), dataset_path)

    total_bindings = 0
    for cluster in CONVERSATIONAL_PAIRS:
        answer = cluster["answer"]
        for query in cluster["inputs"]:
            # 1. Multi-pool train_pair
            mp_payload = json.dumps({
                "src_pool": "in",
                "src": query,
                "tgt_pool": "out",
                "tgt": answer,
                "passes": 40
            }).encode()
            req = urllib.request.Request(
                f"{endpoint}/multi_pool/train_pair",
                data=mp_payload,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    pass
            except Exception as e:
                logger.warning("multi_pool train_pair error for %s: %s", query, e)

            # 2. Slow-pool sequence training
            sp_payload = json.dumps({
                "question": query,
                "answer": answer,
                "lr": 0.45,
                "repeats": 5
            }).encode()
            req_sp = urllib.request.Request(
                f"{endpoint}/train",
                data=sp_payload,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req_sp, timeout=5) as resp:
                    pass
            except Exception as e:
                logger.warning("slow-pool train error for %s: %s", query, e)

            # 3. Dopamine LTP reinforcement
            rf_payload = json.dumps({
                "cue": query,
                "reward": 1.0,
                "dopamine": 0.95
            }).encode()
            req_rf = urllib.request.Request(
                f"{endpoint}/neuro/reinforce",
                data=rf_payload,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req_rf, timeout=5) as resp:
                    pass
            except Exception:
                pass

            total_bindings += 1

    logger.info("Ingested and consolidated %d conversational variations into neural multi-pool", total_bindings)

    # Verify query for "hello"
    test_payload = json.dumps({"text": "Hello", "session_id": "conv-test", "hops": 2, "min_strength": 0.05}).encode()
    req_test = urllib.request.Request(
        f"{endpoint}/chat",
        data=test_payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req_test, timeout=5) as resp:
            res = json.loads(resp.read().decode())
            logger.info("Verification query 'Hello': decoder=%s, answer=%s", res.get("decoder"), res.get("answer")[:60] if res.get("answer") else "")
    except Exception as e:
        logger.warning("Verification error: %s", e)

if __name__ == "__main__":
    ingest_conversational_basics()
