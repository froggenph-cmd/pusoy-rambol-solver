import streamlit as st
from itertools import combinations
from collections import Counter
import random
import json
import os

st.set_page_config(
    page_title="Pusoy Rambol Solver & Learner",
    page_icon="🃏",
    layout="wide",
)

st.title("🃏 Pusoy Rambol Solver & Self-Play Learner")
st.caption(
    "Four-player simulations | Player 1 learning | A♦ tie-break"
)

RANKS = "23456789TJQKA"
SUITS = "CDHS"
RV = {r: i + 2 for i, r in enumerate(RANKS)}

SUIT_SYMBOL = {
    "C": "♣", "D": "♦", "H": "♥", "S": "♠"
}

RANK_LABEL = {
    **{str(i): str(i) for i in range(2, 10)},
    "T": "10", "J": "J", "Q": "Q",
    "K": "K", "A": "A",
}

DECK = [r + s for r in RANKS for s in SUITS]
MODEL_PATH = "pusoy_rambol_model.json"

DEFAULT_WEIGHTS = {
    "bottom": 1.0,
    "middle": 1.0,
    "top": 1.0,
    "top_kicker": 0.08,
    "balance": 0.18,
    "category": 0.02,
}


def card_label(card):
    return RANK_LABEL[card[0]] + SUIT_SYMBOL[card[1]]


def show_cards(cards):
    return "  ".join(card_label(c) for c in cards)


def straight_high(values):
    vals = sorted(set(values))

    if len(vals) != 5:
        return None

    if vals == [2, 3, 4, 5, 14]:
        return 5

    if vals[-1] - vals[0] == 4:
        return vals[-1]

    return None


def rank_five(cards):
    """Rank five-card hands using the user's custom ranking."""
    vals = [RV[c[0]] for c in cards]
    counts = Counter(vals)

    groups = sorted(
        ((n, v) for v, n in counts.items()),
        reverse=True,
    )

    flush = len({c[1] for c in cards}) == 1
    sh = straight_high(vals)

    if flush and sh:
        return (8, sh)

    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(v for v in vals if v != quad)
        return (7, quad, kicker)

    if sorted(counts.values()) == [2, 3]:
        trip = max(v for v, n in counts.items() if n == 3)
        pair = max(v for v, n in counts.items() if n == 2)
        return (6, trip, pair)

    # Custom rule: Flush beats Straight.
    if flush:
        return (5, *sorted(vals, reverse=True))

    if sh:
        return (4, sh)

    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = sorted(
            (v for v in vals if v != trip),
            reverse=True,
        )
        return (3, trip, *kickers)

    pairs = sorted(
        (v for v, n in counts.items() if n == 2),
        reverse=True,
    )

    if len(pairs) == 2:
        kicker = max(v for v in vals if v not in pairs)
        return (2, pairs[0], pairs[1], kicker)

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted(
            (v for v in vals if v != pair),
            reverse=True,
        )
        return (1, pair, *kickers)

    return (0, *sorted(vals, reverse=True))


def rank_top(cards):
    """Top has only Three of a Kind, Pair, or High Card."""
    vals = [RV[c[0]] for c in cards]
    counts = Counter(vals)

    groups = sorted(
        ((n, v) for v, n in counts.items()),
        reverse=True,
    )

    if groups[0][0] == 3:
        return (3, groups[0][1])

    if groups[0][0] == 2:
        pair = groups[0][1]
        kicker = max(v for v in vals if v != pair)
        return (1, pair, kicker)

    return (0, *sorted(vals, reverse=True))


def category(rank, top=False):
    if top:
        return {
            3: "Three of a kind",
            1: "One pair",
            0: "High card",
        }.get(rank[0], "High card")

    names = [
        "High card",
        "One pair",
        "Two pair",
        "Three of a kind",
        "Straight",
        "Flush",
        "Full house",
        "Four of a kind",
        "Straight flush",
    ]

    return names[rank[0]]


def foul_free(bottom, middle, top):
    """
    Valid arrangement:
    Bottom >= Middle >= Top.

    The cross-row comparison uses the category ordering
    currently encoded in this app.
    """
    rb = rank_five(bottom)
    rm = rank_five(middle)
    rt = rank_top(top)

    return rb >= rm and rm[0] >= rt[0]


def tie_value(rank):
    """Approximate numeric strength for heuristic search."""
    value = rank[0] * 100000.0

    for i, x in enumerate(rank[1:]):
        value += x * (1000.0 / (15 ** i))

    return value


def arrangement_candidates(hand, sample_count=450, rng=None):
    """
    Generate valid arrangements.

    sample_count=None performs exhaustive enumeration.
    A numeric sample_count uses random candidates plus
    additional candidates from promising bottom hands.
    """
    rng = rng or random
    hand = tuple(hand)
    found = {}

    def add_candidate(bottom, middle, top):
        rb = rank_five(bottom)
        rm = rank_five(middle)
        rt = rank_top(top)

        if rb < rm or rm[0] < rt[0]:
            return

        key = (
            tuple(sorted(bottom)),
            tuple(sorted(middle)),
            tuple(sorted(top)),
        )

        found[key] = (
            tuple(bottom),
            tuple(middle),
            tuple(top),
            rb,
            rm,
            rt,
        )

    if sample_count is None:
        for bottom in combinations(hand, 5):
            bottom_set = set(bottom)
            rest = tuple(c for c in hand if c not in bottom_set)

            for middle in combinations(rest, 5):
                middle_set = set(middle)
                top = tuple(c for c in rest if c not in middle_set)

                add_candidate(bottom, middle, top)

    else:
        for _ in range(sample_count):
            perm = list(hand)
            rng.shuffle(perm)

            add_candidate(
                tuple(perm[:5]),
                tuple(perm[5:10]),
                tuple(perm[10:13]),
            )

        target = sample_count + 180

        for bottom in combinations(hand, 5):
            rb = rank_five(bottom)

            if rb[0] < 2:
                continue

            bottom_set = set(bottom)
            rest = tuple(c for c in hand if c not in bottom_set)

            for middle in combinations(rest, 5):
                rm = rank_five(middle)

                if rb < rm:
                    continue

                middle_set = set(middle)
                top = tuple(c for c in rest if c not in middle_set)

                if rm[0] < rank_top(top)[0]:
                    continue

                add_candidate(bottom, middle, top)

                if len(found) >= target:
                    break

            if len(found) >= target:
                break

    return list(found.values())


def strategy_score(item, weights):
    bottom, middle, top, rb, rm, rt = item

    values = [
        tie_value(rb) / 1000.0,
        tie_value(rm) / 1000.0,
        tie_value(rt) / 1000.0,
    ]

    score = (
        weights["bottom"] * values[0]
        + weights["middle"] * values[1]
        + weights["top"] * values[2]
    )

    score += weights["category"] * (rb[0] + rm[0] + rt[0])
    score += weights["balance"] * min(values)

    top_values = sorted(
        (RV[c[0]] for c in top),
        reverse=True,
    )

    score += weights["top_kicker"] * sum(top_values)

    return score
def choose_arrangement(hand, weights, mode="learned", rng=None):
    rng = rng or random
    candidates = arrangement_candidates(
        hand,
        sample_count=450,
        rng=rng,
    )

    if not candidates:
        candidates = arrangement_candidates(hand, sample_count=None)

    if not candidates:
        return None

    if mode == "random":
        return rng.choice(candidates)

    if mode == "strong":
        weights = {
            "bottom": 1.0,
            "middle": 1.0,
            "top": 1.0,
            "top_kicker": 0.01,
            "balance": 0.25,
            "category": 0.04,
        }

    elif mode == "balanced":
        weights = {
            "bottom": 0.85,
            "middle": 1.2,
            "top": 1.0,
            "top_kicker": 0.03,
            "balance": 0.4,
            "category": 0.03,
        }

    return max(
        candidates,
        key=lambda item: strategy_score(item, weights),
    )


def head_to_head(arr_a, arr_b, hand_a, hand_b):
    """Return 1 for A's win, -1 for B's win, or 0 for a draw."""
    wins_a = 0
    wins_b = 0

    for rank_a, rank_b in zip(arr_a[3:], arr_b[3:]):
        comparison = (rank_a > rank_b) - (rank_a < rank_b)

        # A♦ breaks an otherwise-equal row tie.
        if comparison == 0:
            a_has_ad = "AD" in hand_a
            b_has_ad = "AD" in hand_b

            if a_has_ad != b_has_ad:
                comparison = 1 if a_has_ad else -1

        if comparison > 0:
            wins_a += 1
        elif comparison < 0:
            wins_b += 1

    if wins_a >= 2:
        return 1

    if wins_b >= 2:
        return -1

    return 0


def deal_game(rng):
    deck = DECK[:]
    rng.shuffle(deck)

    hands = [
        deck[i * 13:(i + 1) * 13]
        for i in range(4)
    ]

    assert all(len(hand) == 13 for hand in hands)
    assert len(set(card for hand in hands for card in hand)) == 52
    assert sum("AD" in hand for hand in hands) == 1

    return hands


def play_deal(hands, weights, deal_seed):
    opponent_modes = ("random", "strong", "balanced")
    arrangements = []

    for i, hand in enumerate(hands):
        rng = random.Random(deal_seed * 1009 + i * 9176 + 31)

        if i == 0:
            mode = "learned"
            use_weights = weights
        else:
            mode = opponent_modes[i - 1]
            use_weights = DEFAULT_WEIGHTS

        arrangement = choose_arrangement(
            hand,
            use_weights,
            mode,
            rng,
        )

        if arrangement is None:
            raise RuntimeError(
                f"No valid arrangement found for Player {i + 1}."
            )

        arrangements.append(arrangement)

    results = [
        head_to_head(
            arrangements[0],
            arrangements[j],
            hands[0],
            hands[j],
        )
        for j in range(1, 4)
    ]

    ad_holder = next(
        i for i, hand in enumerate(hands)
        if "AD" in hand
    )

    return {
        "results": results,
        "p1_has_ad": ad_holder == 0,
        "ad_holder": ad_holder,
        "vs_ad_holder": (
            None if ad_holder == 0 else results[ad_holder - 1]
        ),
    }


def make_deals(number, seed):
    """Create reproducible deals for paired policy comparisons."""
    rng = random.Random(seed)
    deals = []

    for _ in range(number):
        hands = deal_game(rng)
        deal_seed = rng.randrange(1, 2**31 - 1)
        deals.append((hands, deal_seed))

    return deals


def evaluate(weights, deals):
    ad_games = 0
    ad_wins = 0
    no_ad_games = 0
    no_ad_wins = 0
    total_wins = 0
    draws = 0
    losses = 0

    for hands, deal_seed in deals:
        result = play_deal(hands, weights, deal_seed)
        outcomes = result["results"]

        wins = sum(outcome == 1 for outcome in outcomes)
        total_wins += wins
        draws += sum(outcome == 0 for outcome in outcomes)
        losses += sum(outcome == -1 for outcome in outcomes)

        if result["p1_has_ad"]:
            ad_games += 1
            ad_wins += wins
        else:
            no_ad_games += 1

            if result["vs_ad_holder"] == 1:
                no_ad_wins += 1

    # Fraction of all individual matchups won when P1 has A♦.
    ad_rate = (
        ad_wins / (3 * ad_games)
        if ad_games else 0.0
    )

    # Fraction of games won specifically against the A♦ holder
    # when Player 1 does not have A♦.
    vs_ad_rate = (
        no_ad_wins / no_ad_games
        if no_ad_games else 0.0
    )

    overall = (
        total_wins / (3 * len(deals))
        if deals else 0.0
    )

    # Weight the two requested objectives heavily.
    score = (
        0.45 * ad_rate
        + 0.45 * vs_ad_rate
        + 0.10 * overall
    )

    return {
        "score": score,
        "ad_games": ad_games,
        "ad_rate": ad_rate,
        "no_ad_games": no_ad_games,
        "vs_ad_rate": vs_ad_rate,
        "avg_wins": (
            total_wins / len(deals) if deals else 0.0
        ),
        "overall_match_win_rate": overall,
        "draws": draws,
        "losses": losses,
        "games": len(deals),
    }


def normalize_weights(weights):
    normalized = {}

    for key, default in DEFAULT_WEIGHTS.items():
        try:
            value = float(weights.get(key, default))
        except (TypeError, ValueError):
            value = default

        normalized[key] = max(0.001, min(3.0, value))

    return normalized


def initial_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data.get("weights"), dict):
                data["weights"] = normalize_weights(data["weights"])
                data.setdefault("games", 0)
                data.setdefault("history", [])
                data.setdefault("version", 3)
                return data

        except Exception:
            pass

    return {
        "weights": DEFAULT_WEIGHTS.copy(),
        "games": 0,
        "history": [],
        "version": 3,
    }


def train(model, games, seed):
    rng = random.Random(seed)
    current = normalize_weights(model["weights"])

    training_games = max(20, int(games))
    test_games = max(50, min(200, int(games)))

    # Candidate policies share identical training deals.
    training_deals = make_deals(
        training_games,
        seed + 100,
    )

    # A separate batch is used to compare the proposed policy.
    testing_deals = make_deals(
        test_games,
        seed + 900000,
    )

    candidates = [current.copy()]

    for _ in range(4):
        mutant = current.copy()

        for key in mutant:
            if rng.random() < 0.65:
                mutant[key] = max(
                    0.001,
                    min(
                        3.0,
                        mutant[key] * rng.uniform(0.72, 1.32),
                    ),
                )

        candidates.append(mutant)

    training_reports = [
        evaluate(candidate, training_deals)
        for candidate in candidates
    ]

    best_index = max(
        range(len(training_reports)),
        key=lambda i: training_reports[i]["score"],
    )

    proposed = candidates[best_index]

    # Compare current and proposed policies on the same test deals.
    before = evaluate(current, testing_deals)
    proposed_test = evaluate(proposed, testing_deals)

    accepted = proposed_test["score"] > before["score"]
    chosen = proposed if accepted else current
    after = proposed_test if accepted else before

    model["weights"] = normalize_weights(chosen)
    model["games"] = (
        int(model.get("games", 0)) + training_games
    )

    history = model.setdefault("history", [])

    history.append({
        "total_games": model["games"],
        "batch_games": training_games,
        "accepted": accepted,
        "train_score": round(
            training_reports[best_index]["score"], 4
        ),
        "test_before": round(before["score"], 4),
        "test_after": round(after["score"], 4),
        "ad_rate": round(after["ad_rate"], 4),
        "vs_ad_rate": round(after["vs_ad_rate"], 4),
        "avg_wins": round(after["avg_wins"], 3),
        "test_games": test_games,
    })

    model["history"] = history[-100:]

    return model, {
        "accepted": accepted,
        "before": before,
        "after": after,
        "training_games": training_games,
        "test_games": test_games,
    }


def save_model(model):
    with open(MODEL_PATH, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)
# ============================================================
# PART 3: STREAMLIT INTERFACE
# ============================================================

# Load the model once per session.
if "model_state" not in st.session_state:
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "r", encoding="utf-8") as file:
                loaded_model = json.load(file)

            if isinstance(loaded_model, dict) and "weights" in loaded_model:
                st.session_state.model_state = loaded_model
            else:
                st.session_state.model_state = initial_model()
        else:
            st.session_state.model_state = initial_model()
    except Exception:
        st.session_state.model_state = initial_model()

model = st.session_state.model_state
model.setdefault("weights", DEFAULT_WEIGHTS.copy())
model.setdefault("history", [])
model.setdefault("training_deals", 0)
model.setdefault("latest_test_score", None)
model.setdefault("latest_result", "No training completed yet.")

weights = normalize_weights(model["weights"])

st.title("🃏 Pusoy Rambol Solver & Learner")
st.caption(
    "Arrange your 13 cards, compare strategies, and train the solver "
    "using your Pusoy Rambol rules."
)

solver_tab, learner_tab, rules_tab = st.tabs(
    ["🂡 Hand Solver", "🧠 Learner & Backup", "📘 Rules"]
)


# ------------------------------------------------------------
# TAB 1: HAND SOLVER
# ------------------------------------------------------------
with solver_tab:
    st.subheader("Enter your 13 cards")
    st.write(
        "Choose every card in your hand. Each card must be unique. "
        "The solver will search for foul-free arrangements."
    )

    card_options = [card_label(card) for card in DECK]

    # Store the card selections as labels, then convert them to card IDs.
    selected_labels = []

    with st.form("hand_solver_form"):
        st.write("**Your cards**")

        for row_start in range(0, 13, 4):
            cols = st.columns(4)

            for offset, col in enumerate(cols):
                card_index = row_start + offset

                if card_index >= 13:
                    break

                default_label = card_options[card_index]

                with col:
                    chosen = st.selectbox(
                        f"Card {card_index + 1}",
                        options=card_options,
                        index=card_index,
                        key=f"hand_card_{card_index}",
                    )

                selected_labels.append(chosen)

        solve_clicked = st.form_submit_button(
            "Find my best arrangements",
            type="primary",
            use_container_width=True,
        )

    if solve_clicked:
        if len(set(selected_labels)) != 13:
            st.error(
                "Your hand contains duplicate cards. "
                "Please select 13 different cards."
            )
        else:
            label_to_card = {
                card_label(card): card for card in DECK
            }
            hand = [label_to_card[label] for label in selected_labels]

            with st.spinner("Searching for arrangements..."):
                candidates = arrangement_candidates(
                    hand,
                    sample_count=None,
                )

                if not candidates:
                    st.error(
                        "No valid arrangement was found. "
                        "Please check the foul rules in Part 1."
                    )
                else:
                    # Rank candidate arrangements using the current model.
                    ranked = sorted(
                        candidates,
                        key=lambda arrangement: strategy_score(
                            arrangement, weights
                        ),
                        reverse=True,
                    )

                    # Remove duplicate strength profiles from the display
                    # so the suggestions are more varied.
                    displayed = []
                    seen_profiles = set()

                    for arrangement in ranked:
                        bottom, middle, top, rb, rm, rt = arrangement

                        profile = (
                            tuple(rb),
                            tuple(rm),
                            tuple(rt),
                        )

                        if profile in seen_profiles:
                            continue

                        seen_profiles.add(profile)
                        displayed.append(arrangement)

                        if len(displayed) >= 10:
                            break

            st.success(
                f"Found {len(candidates):,} candidate arrangements."
            )

            st.caption(
                "Suggestions are ranked by the current learned scoring "
                "weights. They are not guaranteed to be mathematically "
                "optimal against every possible opponent."
            )

            for number, arrangement in enumerate(displayed, start=1):
                bottom, middle, top, rb, rm, rt = arrangement
                score = strategy_score(arrangement, weights)

                with st.container(border=True):
                    st.markdown(f"### Suggestion {number}")
                    st.caption(f"Model score: {score:.3f}")

                    st.markdown("**TOP — 3 cards**")
                    st.write(show_cards(top))

                    st.markdown("**MIDDLE — 5 cards**")
                    st.write(show_cards(middle))

                    st.markdown("**BOTTOM — 5 cards**")
                    st.write(show_cards(bottom))

                    st.caption(
                        "Rank categories — "
                        f"Bottom: {category(rb)}, "
                        f"Middle: {category(rm)}, "
                        f"Top: {category(rt)}"
                    )

                    if number == 1:
                        st.info(
                            "This is the model's highest-scoring "
                            "suggestion for the current weights."
                        )


# ------------------------------------------------------------
# TAB 2: LEARNER, MODEL CHECKPOINTS, AND BACKUP
# ------------------------------------------------------------
with learner_tab:
    st.subheader("Train the strategy model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training deals",
            f"{int(model.get('training_deals', 0)):,}",
        )

    with col2:
        latest_score = model.get("latest_test_score")
        score_display = (
            "Not tested"
            if latest_score is None
            else f"{float(latest_score):.3f}"
        )
        st.metric("Latest test score", score_display)

    with col3:
        st.metric(
            "Training rounds",
            f"{len(model.get('history', [])):,}",
        )

    st.write(model.get("latest_result", "No training completed yet."))

    st.info(
        "Training compares a proposed set of scoring weights with the "
        "current weights using simulated deals. Results depend on the "
        "opponent strategies and random deals used. A higher score on "
        "these simulations does not prove the strategy is universally best."
    )

    training_games = st.select_slider(
        "Number of training deals",
        options=[25, 50, 100, 250, 500],
        value=50,
        key="training_games",
    )

    training_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=20261028,
        step=1,
        key="training_seed",
    )

    if st.button(
        "Train model",
        type="primary",
        use_container_width=True,
        key="train_model_button",
    ):
        with st.spinner(
            f"Training with {training_games} simulated deals..."
        ):
            try:
                old_model = st.session_state.model_state

                updated_model = train(
                    old_model,
                    num_deals=int(training_games),
                    seed=int(training_seed),
                )

                st.session_state.model_state = updated_model

                try:
                    save_model(updated_model)
                    st.success(
                        "Training finished and the model checkpoint "
                        "was saved in the app's current environment."
                    )
                except Exception as save_error:
                    st.warning(
                        "Training finished, but the checkpoint could "
                        f"not be saved: {save_error}"
                    )

                st.rerun()

            except Exception as train_error:
                st.error(f"Training failed: {train_error}")

    st.divider()
    st.subheader("Training history")

    history = model.get("history", [])

    if history:
        for entry in reversed(history[-10:]):
            with st.expander(
                f"Round {entry.get('round', '?')} — "
                f"{entry.get('result', 'No result')}"
            ):
                st.write(
                    f"Deals: {entry.get('deals', 'N/A')}"
                )
                st.write(
                    f"Score before: {entry.get('before_score', 'N/A')}"
                )
                st.write(
                    f"Score after: {entry.get('after_score', 'N/A')}"
                )
                st.write(
                    f"Accepted: {entry.get('accepted', 'N/A')}"
                )
    else:
        st.caption("No training history yet. Train the model to begin.")

    st.divider()
    st.subheader("Current scoring weights")

    current_weights = normalize_weights(model["weights"])

    for weight_name, weight_value in current_weights.items():
        st.write(f"**{weight_name.replace('_', ' ').title()}**")
        st.progress(
            min(1.0, max(0.0, float(weight_value))),
            text=f"{float(weight_value):.4f}",
        )

    st.divider()
    st.subheader("Download a model backup")

    checkpoint_json = json.dumps(model, indent=2, ensure_ascii=False)

    st.download_button(
        label="Download model checkpoint (JSON)",
        data=checkpoint_json,
        file_name="pusoy_rambol_model.json",
        mime="application/json",
        use_container_width=True,
    )

    st.caption(
        "Keep this JSON file somewhere safe. Streamlit Community Cloud "
        "may not preserve files written by the app after a restart or "
        "redeployment."
    )

    st.divider()
    st.subheader("Restore a model checkpoint")

    uploaded_checkpoint = st.file_uploader(
        "Choose a previously downloaded model JSON file",
        type=["json"],
        key="checkpoint_uploader",
    )

    if st.button(
        "Restore uploaded checkpoint",
        use_container_width=True,
        disabled=uploaded_checkpoint is None,
        key="restore_checkpoint_button",
    ):
        try:
            restored_model = json.load(uploaded_checkpoint)

            if not isinstance(restored_model, dict):
                raise ValueError("The JSON root must be an object.")

            if "weights" not in restored_model:
                raise ValueError("The file does not contain model weights.")

            if not isinstance(restored_model["weights"], dict):
                raise ValueError("The model weights must be an object.")

            restored_model["weights"] = normalize_weights(
                restored_model["weights"]
            )
            restored_model.setdefault("history", [])
            restored_model.setdefault("training_deals", 0)
            restored_model.setdefault("latest_test_score", None)
            restored_model.setdefault(
                "latest_result",
                "Restored from checkpoint.",
            )

            st.session_state.model_state = restored_model

            try:
                save_model(restored_model)
            except Exception:
                pass

            st.success("Checkpoint loaded into the current session.")
            st.rerun()

        except Exception as restore_error:
            st.error(
                f"Could not restore this checkpoint: {restore_error}"
            )

    st.divider()
    st.subheader("Reset model")

    st.warning(
        "Resetting replaces the current learned weights and training "
        "history with the default model. Download a backup first if "
        "you want to keep your progress."
    )

    confirm_reset = st.checkbox(
        "I understand that resetting removes the current session's "
        "learned model and history.",
        key="confirm_reset_model",
    )

    if st.button(
        "Reset to default model",
        disabled=not confirm_reset,
        use_container_width=True,
        key="reset_model_button",
    ):
        fresh_model = initial_model()
        st.session_state.model_state = fresh_model

        try:
            save_model(fresh_model)
        except Exception:
            pass

        st.success("The default model is loaded.")
        st.rerun()


# ------------------------------------------------------------
# TAB 3: RULES AND MODEL NOTES
# ------------------------------------------------------------
with rules_tab:
    st.subheader("Pusoy Rambol rules used by this app")

    st.markdown(
        """
        **Deal**
        - A standard 52-card deck is used.
        - Four players receive 13 cards each.
        - Each player's 13 cards are arranged exactly once into:
          Bottom (5), Middle (5), and Top (3).

        **Five-card hand rankings, strongest to weakest**
        1. Straight Flush
        2. Four of a Kind
        3. Full House
        4. Flush
        5. Straight
        6. Three of a Kind
        7. Two Pair
        8. One Pair
        9. High Card

        **Top three-card hand rankings**
        1. Three of a Kind
        2. Pair
        3. High Card

        Straights and flushes do not apply to the Top hand.

        **Foul rule**
        - Bottom must be stronger than or equal to Middle.
        - Middle must be stronger than or equal to Top.
        - Equal strength is allowed under the rule you specified.

        **Head-to-head scoring**
        - Bottom is compared with Bottom.
        - Middle is compared with Middle.
        - Top is compared with Top.
        - Winning at least two of the three rows wins the head-to-head.
        - If neither player wins at least two rows, the result is a draw.

        **A♦ tie-break**
        - When two compared rows tie, the app checks whether a player
          holds A♦ anywhere in their 13-card hand.
        - This is your special rule and is separate from ordinary
          poker hand rankings.
        """
    )

    st.divider()
    st.subheader("Important model limitations")

    st.markdown(
        """
        - The learner improves its scoring weights through simulated
          games; it does not prove that an arrangement is globally optimal.
        - The hand solver's suggestions depend on the candidate search
          and current model weights.
        - Training performance on simulated deals may not predict
          performance against every real opponent.
        - A downloaded JSON checkpoint is the safest way to preserve
          training progress between Streamlit Cloud restarts.
        """
    )

    st.caption(
        "If a result seems inconsistent with your table rules, "
        "record the 13 cards and the suggested rows so the ranking "
        "or comparison logic can be checked."
    )

st.divider()
st.caption(
    "Pusoy Rambol Solver — personal strategy-learning project."
)