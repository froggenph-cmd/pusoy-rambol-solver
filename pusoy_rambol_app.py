import streamlit as st
from itertools import combinations
from collections import Counter
from dataclasses import dataclass

st.set_page_config(page_title="Pusoy Rambol Solver", page_icon="🃏", layout="wide")
st.title("🃏 Pusoy Rambol Solver")
st.caption("Enter your 13 cards to find strong, foul-free 5–5–3 arrangements.")

RANKS = "23456789TJQKA"
SUITS = "CDHS"
RV = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_SYMBOL = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
RANK_LABEL = {**{str(i): str(i) for i in range(2, 10)}, "T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}
DECK = [r+s for r in RANKS for s in SUITS]

def card_label(c):
    return f"{RANK_LABEL[c[0]]}{SUIT_SYMBOL[c[1]]}"

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
    vals = [RV[c[0]] for c in cards]
    counts = Counter(vals)
    groups = sorted(((n, v) for v, n in counts.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    sh = straight_high(vals)
    # User's custom order: straight flush > quads > full house > flush > straight > trips > two pair > pair > high card
    if flush and sh: return (8, sh)
    if groups[0][0] == 4:
        q = groups[0][1]
        return (7, q, max(v for v in vals if v != q))
    if sorted(counts.values()) == [2, 3]:
        return (6, max(v for v,n in counts.items() if n == 3),
                max(v for v,n in counts.items() if n == 2))
    if flush: return (5, *sorted(vals, reverse=True))
    if sh: return (4, sh)
    if groups[0][0] == 3:
        t = groups[0][1]
        return (3, t, *sorted((v for v in vals if v != t), reverse=True))
    pairs = sorted((v for v,n in counts.items() if n == 2), reverse=True)
    if len(pairs) == 2:
        return (2, pairs[0], pairs[1], max(v for v in vals if v not in pairs))
    if len(pairs) == 1:
        p = pairs[0]
        return (1, p, *sorted((v for v in vals if v != p), reverse=True))
    return (0, *sorted(vals, reverse=True))

def rank_top(cards):
    vals = [RV[c[0]] for c in cards]
    counts = Counter(vals)
    groups = sorted(((n, v) for v,n in counts.items()), reverse=True)
    if groups[0][0] == 3: return (3, groups[0][1])
    if groups[0][0] == 2:
        p = groups[0][1]
        return (1, p, max(v for v in vals if v != p))
    return (0, *sorted(vals, reverse=True))

def category(rank, top=False):
    names = ["High card", "One pair", "Two pair", "Three of a kind", "Straight",
             "Flush", "Full house", "Four of a kind", "Straight flush"]
    if top:
        return {0:"High card", 1:"One pair", 3:"Three of a kind"}.get(rank[0], "High card")
    return names[rank[0]]

def foul_free(bottom, middle, top):
    rb, rm, rt = rank_five(bottom), rank_five(middle), rank_top(top)
    # For foul checking, compare the strength category ordering across 5-card and 3-card hands.
    # Top trips is stronger than a middle three-of-a-kind only if category rank is equal; equal category allowed.
    return rb >= rm and rm[0] >= rt[0]

def tie_value(rank):
    # Stable scalar used only to rank suggestions, not for head-to-head comparisons.
    val = rank[0] * 100000
    for i, x in enumerate(rank[1:]):
        val += x * (1000 / (15 ** i))
    return val

def generate(hand, limit=10):
    found = []
    hand = tuple(hand)
    for b in combinations(hand, 5):
        rest = tuple(c for c in hand if c not in b)
        rb = rank_five(b)
        for m in combinations(rest, 5):
            top = tuple(c for c in rest if c not in m)
            rm, rt = rank_five(m), rank_top(top)
            if rb < rm or rm[0] < rt[0]:
                continue
            # Heuristic for unknown opponents: prioritize having two very strong
            # levels rather than maximizing Bottom alone. This is a suggestion score,
            # not a game-theoretic proof of optimality.
            score = 0.38*tie_value(rb) + 0.37*tie_value(rm) + 0.25*tie_value(rt)
            found.append((score, b, m, top, rb, rm, rt))
    found.sort(key=lambda x: x[0], reverse=True)
    return found[:limit], len(found)

def show_cards(cards):
    return "  ".join(card_label(c) for c in cards)

st.info("Choose all 13 cards. Each card can be selected only once. The app will show strong legal arrangements, ranked by a balanced heuristic—not a proven game-theoretic optimum against unknown opponents.")
with st.form("cards_form"):
    cols = st.columns(4)
    selected = []
    for i in range(13):
        with cols[i % 4]:
            card = st.selectbox(
                f"Card {i+1}",
                options=["— choose —"] + DECK,
                format_func=lambda c: c if c == "— choose —" else card_label(c),
                key=f"card_{i}",
            )
            selected.append(card)
    submitted = st.form_submit_button("Find strong arrangements", type="primary", use_container_width=True)

if submitted:
    cards = [c for c in selected if c != "— choose —"]
    if len(cards) != 13:
        st.error(f"Please select all 13 cards. You have selected {len(cards)}.")
    elif len(set(cards)) != 13:
        st.error("A card was selected more than once. Each card must be unique.")
    else:
        with st.spinner("Checking legal arrangements..."):
            results, count = generate(cards)
        st.write(f"**{count:,} legal arrangements** found and checked.")
        if not results:
            st.error("No legal arrangements were found. Please check the foul rules or your cards.")
        else:
            for idx, item in enumerate(results, start=1):
                score, b, m, t, rb, rm, rt = item
                with st.container(border=True):
                    st.subheader(f"Option {idx}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**BOTTOM — 5 cards**")
                        st.write(show_cards(b))
                        st.caption(category(rb))
                    with c2:
                        st.markdown("**MIDDLE — 5 cards**")
                        st.write(show_cards(m))
                        st.caption(category(rm))
                    with c3:
                        st.markdown("**TOP — 3 cards**")
                        st.write(show_cards(t))
                        st.caption(category(rt, top=True))
                    st.caption(f"Suggestion score: {score:,.0f} · This is a heuristic ranking, not a guaranteed win rate.")
            st.warning("A♦ tie-break applies only when the compared levels are otherwise tied and one player holds A♦ anywhere in their 13 cards. Because opponents' cards are unknown here, the app cannot calculate an exact strategic equilibrium.")
else:
    st.markdown("### How to use")
    st.markdown("1. Select each of your 13 cards from the dropdowns.\n2. Press **Find strong arrangements**.\n3. Compare the suggested Bottom, Middle and Top hands. The app shows up to 10 legal options.")
