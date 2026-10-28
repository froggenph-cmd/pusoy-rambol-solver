import streamlit as st
from itertools import combinations
from collections import Counter
import random, json, os, math

st.set_page_config(page_title="Pusoy Rambol Solver & Learner", page_icon="🃏", layout="wide")
st.title("🃏 Pusoy Rambol Solver & Self-Play Learner")
st.caption("Custom Pusoy Rambol rules · four-player simulations · Player 1 learning")

RANKS = "23456789TJQKA"
SUITS = "CDHS"
RV = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_SYMBOL = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
RANK_LABEL = {**{str(i): str(i) for i in range(2, 10)}, "T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}
DECK = [r+s for r in RANKS for s in SUITS]
MODEL_PATH = "pusoy_rambol_model.json"
DEFAULT_WEIGHTS = {"bottom": 1.0, "middle": 1.0, "top": 1.0, "top_kicker": 0.08, "balance": 0.18, "category": 0.02}


def card_label(c): return f"{RANK_LABEL[c[0]]}{SUIT_SYMBOL[c[1]]}"
def show_cards(cards): return "  ".join(card_label(c) for c in cards)
def straight_high(values):
    vals = sorted(set(values))
    if len(vals) != 5: return None
    if vals == [2, 3, 4, 5, 14]: return 5
    if vals[-1] - vals[0] == 4: return vals[-1]
    return None

def rank_five(cards):
    vals = [RV[c[0]] for c in cards]
    counts = Counter(vals)
    groups = sorted(((n, v) for v, n in counts.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    sh = straight_high(vals)
    # Custom order: straight flush > four-kind > full house > flush > straight > trips > two pair > pair > high card.
    if flush and sh: return (8, sh)
    if groups[0][0] == 4:
        q = groups[0][1]; return (7, q, max(v for v in vals if v != q))
    if sorted(counts.values()) == [2, 3]:
        return (6, max(v for v,n in counts.items() if n == 3), max(v for v,n in counts.items() if n == 2))
    if flush: return (5, *sorted(vals, reverse=True))
    if sh: return (4, sh)
    if groups[0][0] == 3:
        t = groups[0][1]; return (3, t, *sorted((v for v in vals if v != t), reverse=True))
    pairs = sorted((v for v,n in counts.items() if n == 2), reverse=True)
    if len(pairs) == 2: return (2, pairs[0], pairs[1], max(v for v in vals if v not in pairs))
    if len(pairs) == 1:
        p = pairs[0]; return (1, p, *sorted((v for v in vals if v != p), reverse=True))
    return (0, *sorted(vals, reverse=True))

def rank_top(cards):
    vals = [RV[c[0]] for c in cards]; counts = Counter(vals)
    groups = sorted(((n, v) for v,n in counts.items()), reverse=True)
    if groups[0][0] == 3: return (3, groups[0][1])
    if groups[0][0] == 2:
        p = groups[0][1]; return (1, p, max(v for v in vals if v != p))
    return (0, *sorted(vals, reverse=True))

def category(rank, top=False):
    names = ["High card", "One pair", "Two pair", "Three of a kind", "Straight", "Flush", "Full house", "Four of a kind", "Straight flush"]
    return ({0:"High card", 1:"One pair", 3:"Three of a kind"}.get(rank[0], "High card") if top else names[rank[0]])

def foul_free(bottom, middle, top):
    rb, rm, rt = rank_five(bottom), rank_five(middle), rank_top(top)
    # Current app rule: Bottom must rank at least as high as Middle; Middle category must be >= Top category.
    return rb >= rm and rm[0] >= rt[0]

def tie_value(rank):
    # Numeric proxy for strategy search only; actual comparisons always use rank tuples.
    v = rank[0] * 100000.0
    for i, x in enumerate(rank[1:]): v += x * (1000.0 / (15 ** i))
    return v

def arrangement_candidates(hand, sample_count=800, rng=None):
    """Sample unique partitions, plus exhaustive partitions when sample_count=None."""
    rng = rng or random
    hand = tuple(hand)
    found = {}
    if sample_count is None:
        for b in combinations(hand, 5):
            bset = set(b); rest = tuple(c for c in hand if c not in bset); rb = rank_five(b)
            for m in combinations(rest, 5):
                mset = set(m); top = tuple(c for c in rest if c not in mset)
                rm, rt = rank_five(m), rank_top(top)
                if rb < rm or rm[0] < rt[0]: continue
                key = (tuple(sorted(b)), tuple(sorted(m)), tuple(sorted(top)))
                found[key] = (b, m, top, rb, rm, rt)
    else:
        # Random partitions offer broad coverage at manageable training cost.
        for _ in range(sample_count):
            perm = list(hand); rng.shuffle(perm)
            b, m, top = tuple(perm[:5]), tuple(perm[5:10]), tuple(perm[10:13])
            rb, rm, rt = rank_five(b), rank_five(m), rank_top(top)
            if rb < rm or rm[0] < rt[0]: continue
            key = (tuple(sorted(b)), tuple(sorted(m)), tuple(sorted(top)))
            found[key] = (b, m, top, rb, rm, rt)
        # Add a few deliberate candidate partitions from ranked five-card subsets to avoid relying only on luck.
        for b in combinations(hand, 5):
            rb = rank_five(b)
            if rb[0] < 2: continue
            rest = tuple(c for c in hand if c not in set(b))
            for m in combinations(rest, 5):
                rm = rank_five(m)
                if rb < rm: continue
                top = tuple(c for c in rest if c not in set(m)); rt = rank_top(top)
                if rm[0] < rt[0]: continue
                key = (tuple(sorted(b)), tuple(sorted(m)), tuple(sorted(top)))
                found.setdefault(key, (b, m, top, rb, rm, rt))
                if len(found) >= sample_count + 180: break
            if len(found) >= sample_count + 180: break
    return list(found.values())

def strategy_score(item, weights):
    b,m,t,rb,rm,rt = item
    # Keep the three row-strength contributions distinct; balance rewards avoiding one very weak row.
    vals = [tie_value(rb)/1000.0, tie_value(rm)/1000.0, tie_value(rt)/1000.0]
    base = weights["bottom"]*vals[0] + weights["middle"]*vals[1] + weights["top"]*vals[2]
    base += weights["category"]*(rb[0]+rm[0]+rt[0])
    base += weights["balance"]*min(vals)
    # User's stated preference: high kickers in Top where this does not violate arrangement order.
    top_vals = sorted((RV[c[0]] for c in t), reverse=True)
    base += weights["top_kicker"]*sum(top_vals)
    return base

def choose_arrangement(hand, weights, mode="learned", rng=None):
    rng = rng or random
    candidates = arrangement_candidates(hand, sample_count=450, rng=rng)
    if not candidates:
        # Fallback exhaustive search; should be rare but keeps the simulator honest.
        candidates = arrangement_candidates(hand, sample_count=None, rng=rng)
    if not candidates: return None
    if mode == "random": return rng.choice(candidates)
    if mode == "strong":
        w = {"bottom":1.0,"middle":1.0,"top":1.0,"top_kicker":0.01,"balance":0.25,"category":0.04}
        return max(candidates, key=lambda x: strategy_score(x,w))
    if mode == "balanced":
        w = {"bottom":0.85,"middle":1.2,"top":1.0,"top_kicker":0.03,"balance":0.4,"category":0.03}
        return max(candidates, key=lambda x: strategy_score(x,w))
    return max(candidates, key=lambda x: strategy_score(x,weights))

def compare_rank(a,b): return (a > b) - (a < b)
def head_to_head(arr_a, arr_b, hand_a, hand_b):
    wins_a = 0; wins_b = 0
    for ra, rb in zip(arr_a[3:], arr_b[3:]):
        cmp = compare_rank(ra, rb)
        if cmp == 0:
            a_has = "AD" in hand_a; b_has = "AD" in hand_b
            if a_has != b_has: cmp = 1 if a_has else -1
        if cmp > 0: wins_a += 1
        elif cmp < 0: wins_b += 1
    # A head-to-head win requires winning at least two rows. Exact ties stay ties.
    if wins_a >= 2: return 1
    if wins_b >= 2: return -1
    return 0

def deal_game(rng):
    deck = DECK[:]; rng.shuffle(deck)
    hands = [deck[i*13:(i+1)*13] for i in range(4)]
    assert all(len(h)==13 and len(set(h))==13 for h in hands)
    assert sum("AD" in h for h in hands) == 1
    return hands

def play_game(weights, rng, opponent_modes=("random","strong","balanced"), sample_count_note=None):
    hands = deal_game(rng)
    modes = ["learned", *opponent_modes]
    arrangements = []
    for i,h in enumerate(hands):
        arrangements.append(choose_arrangement(h, weights if i==0 else DEFAULT_WEIGHTS, modes[i], rng))
    results = [head_to_head(arrangements[0], arrangements[j], hands[0], hands[j]) for j in range(1,4)]
    ad_holder = next(i for i,h in enumerate(hands) if "AD" in h)
    player1_has_ad = ad_holder == 0
    vs_ad_holder = None if player1_has_ad else results[ad_holder-1]
    return {"hands":hands,"arrangements":arrangements,"results":results,"p1_has_ad":player1_has_ad,"ad_holder":ad_holder,"vs_ad_holder":vs_ad_holder}

def initial_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH,"r",encoding="utf-8") as f: data=json.load(f)
            if isinstance(data.get("weights"),dict): return data
        except Exception: pass
    return {"weights":DEFAULT_WEIGHTS.copy(),"games":0,"history":[],"version":1}

def normalize_weights(w):
    keys=DEFAULT_WEIGHTS.keys(); out={}
    for k in keys:
        x=float(w.get(k,DEFAULT_WEIGHTS[k])); out[k]=max(0.001,min(3.0,x))
    return out

def train(model, games, seed):
    rng=random.Random(seed)
    weights=normalize_weights(model["weights"])
    # Small evolutionary policy search: compare current policy with nearby variants on the same random deal seeds.
    # Each batch tests current and perturbed policies against the same opponent mix.
    wins_before=0; matches_before=0; ad_games=0; ad_wins=0; no_ad_games=0; no_ad_vs_holder=0; no_ad_holder_wins=0
    candidate_weights=[]
    candidate_weights.append(weights.copy())
    for _ in range(4):
        mutant=weights.copy()
        for k in mutant:
            if rng.random()<0.65: mutant[k]=max(0.001,min(3.0,mutant[k]*rng.uniform(0.72,1.32)))
        candidate_weights.append(mutant)
    # Evaluate a modest shared batch per candidate; the requested game count is the overall training budget.
    per=max(1, games//len(candidate_weights))
    scores=[]; details=[]
    for ci,cw in enumerate(candidate_weights):
        total=0; games_ad=0; wins_ad=0; games_no=0; wins_vs_ad=0; draws=0; outcomes=[]
        for _ in range(per):
            result=play_game(cw,rng)
            w=sum(1 for x in result["results"] if x==1)
            d=sum(1 for x in result["results"] if x==0)
            total += w + 0.25*d
            outcomes.append(w)
            if result["p1_has_ad"]:
                games_ad += 1; wins_ad += w
            else:
                games_no += 1
                if result["vs_ad_holder"] == 1: wins_vs_ad += 1
        # Objective: if P1 has A♦, maximize total H2H wins; otherwise prioritize beating A♦ holder.
        # The two conditional rates get equal importance so neither situation is ignored.
        ad_rate=(wins_ad/(games_ad*3)) if games_ad else 0
        vs_ad_rate=(wins_vs_ad/games_no) if games_no else 0
        avg_total=total/(per*3)
        score=0.45*ad_rate+0.45*vs_ad_rate+0.10*avg_total
        scores.append(score); details.append({"score":score,"games":per,"ad_rate":ad_rate,"vs_ad_rate":vs_ad_rate,"avg_wins":sum(outcomes)/per if outcomes else 0,"weights":cw})
    best_i=max(range(len(scores)),key=lambda i:scores[i])
    best=details[best_i]
    model["weights"]=normalize_weights(best["weights"])
    model["games"] = int(model.get("games",0))+per*len(candidate_weights)
    hist=model.setdefault("history",[])
    hist.append({"total_games":model["games"],"batch_games":per*len(candidate_weights),"score":round(best["score"],4),"ad_rate":round(best["ad_rate"],4),"vs_ad_rate":round(best["vs_ad_rate"],4),"avg_wins":round(best["avg_wins"],3)})
    model["history"]=hist[-100:]
    return model, details

def save_model(model):
    with open(MODEL_PATH,"w",encoding="utf-8") as f: json.dump(model,f,indent=2)

model=initial_model()
if "model_state" not in st.session_state: st.session_state.model_state=model
model=st.session_state.model_state
weights=normalize_weights(model["weights"])

solver_tab, learner_tab, rules_tab = st.tabs(["Hand solver", "Self-play learner", "Rules & model"])
with solver_tab:
    st.subheader("Find strong arrangements for your 13 cards")
    st.info("Choose all 13 unique cards. Suggestions are heuristic, not guaranteed optimal against unknown opponents.")
    with st.form("cards_form"):
        cols=st.columns(4); selected=[]
        for i in range(13):
            with cols[i%4]:
                selected.append(st.selectbox(f"Card {i+1}",options=["— choose —"]+DECK,format_func=lambda c:c if c=="— choose —" else card_label(c),key=f"card_{i}"))
        submitted=st.form_submit_button("Find strong arrangements",type="primary",use_container_width=True)
    if submitted:
        cards=[c for c in selected if c!="— choose —"]
        if len(cards)!=13: st.error(f"Please select all 13 cards. You selected {len(cards)}.")
        elif len(set(cards))!=13: st.error("A card was selected more than once.")
        else:
            with st.spinner("Searching legal arrangements..."):
                results=arrangement_candidates(cards,sample_count=None)
                results.sort(key=lambda x:strategy_score(x,weights),reverse=True)
                # Deduplicate strength-equivalent rows to show materially different options.
                unique=[]; seen=set()
                for item in results:
                    sig=(item[3],item[4],item[5])
                    if sig in seen: continue
                    seen.add(sig); unique.append(item)
                    if len(unique)>=10: break
            st.write(f"Found **{len(results):,} legal arrangements**. Showing up to 10 distinct hand-strength profiles.")
            for idx,item in enumerate(unique,1):
                b,m,t,rb,rm,rt=item
                with st.container(border=True):
                    st.subheader(f"Option {idx}"); c1,c2,c3=st.columns(3)
                    with c1: st.markdown("**BOTTOM · 5**"); st.write(show_cards(b)); st.caption(category(rb))
                    with c2: st.markdown("**MIDDLE · 5**"); st.write(show_cards(m)); st.caption(category(rm))
                    with c3: st.markdown("**TOP · 3**"); st.write(show_cards(t)); st.caption(category(rt,True))
                    st.caption(f"Policy score: {strategy_score(item,weights):,.1f}")

with learner_tab:
    st.subheader("Player 1 self-play training")
    st.write("Player 1 is the main learner. A♦ is shuffled into one of the four hands each game. Opponents use a mix of random, strong-hand, and balanced strategies.")
    m1,m2,m3=st.columns(3)
    m1.metric("Training games",f"{model.get('games',0):,}")
    m2.metric("Current A♦-holder score",f"{(model.get('history') or [{}])[-1].get('ad_rate',0)*100:.1f}%" if model.get("history") else "Not measured")
    m3.metric("Current vs A♦ holder",f"{(model.get('history') or [{}])[-1].get('vs_ad_rate',0)*100:.1f}%" if model.get("history") else "Not measured")
    st.markdown("**Player 1's objectives**")
    st.markdown("- **When Player 1 has A♦:** maximize head-to-head wins against all three opponents.\n- **When Player 1 does not have A♦:** prioritize winning against the player who has A♦.\n- Keep these two outcomes separately measured during learning.")
    n_games=st.select_slider("Training game budget",options=[25,50,100,250,500],value=100,help="Each candidate policy is evaluated on a portion of this budget. Training may take time because legal arrangements are sampled for every hand.")
    seed=st.number_input("Random seed (reproducible run)",min_value=0,max_value=2_147_483_647,value=20261028,step=1)
    if st.button("Train learner",type="primary",use_container_width=True):
        with st.spinner("Simulating games and testing nearby strategies. Keep this tab open..."):
            try:
                new_model,report=train({"weights":weights,"games":model.get("games",0),"history":model.get("history",[]),"version":1},int(n_games),int(seed))
                st.session_state.model_state=new_model
                try: save_model(new_model)
                except Exception: pass
                st.success(f"Training batch completed. Total recorded games: {new_model['games']:,}.")
                st.rerun()
            except Exception as e: st.error(f"Training stopped: {e}")
    if model.get("history"):
        st.markdown("#### Recent training history")
        st.dataframe([{ "Total games":h["total_games"],"Batch games":h["batch_games"],"A♦ holder win fraction":f"{h['ad_rate']*100:.1f}%","Win vs A♦ holder":f"{h['vs_ad_rate']*100:.1f}%","Avg H2H wins/game":h["avg_wins"],"Policy score":h["score"]} for h in model["history"][-10:]],use_container_width=True,hide_index=True)
    st.markdown("#### Current learned weights")
    st.json(weights)
    st.download_button("Download learned model JSON",data=json.dumps(model,indent=2),file_name="pusoy_rambol_model.json",mime="application/json",use_container_width=True)
    if st.button("Reset learner to default",type="secondary"):
        st.session_state.model_state={"weights":DEFAULT_WEIGHTS.copy(),"games":0,"history":[],"version":1}
        try:
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        except Exception: pass
        st.rerun()

with rules_tab:
    st.subheader("Rules currently encoded")
    st.markdown("- Standard 52-card deck; four players receive 13 cards each.\n- A♦ is randomly distributed by shuffling and dealing.\n- Each arrangement uses Bottom 5, Middle 5, Top 3 with no card reused.\n- Five-card ranking: Straight Flush > Four of a Kind > Full House > Flush > Straight > Three of a Kind > Two Pair > One Pair > High Card.\n- Top three-card ranking: Three of a Kind > Pair > High Card. No straight or flush in Top.\n- A head-to-head match is won by winning at least two of the three corresponding rows.\n- A♦ breaks otherwise-equal row ties if exactly one player holds A♦ anywhere in their 13-card hand.\n- Current foul validation requires Bottom rank ≥ Middle rank and Middle category ≥ Top category.")
    st.warning("The cross-row foul rule is currently the app's existing simplified rule. Please confirm if your table uses a different exact comparison between 5-card and 3-card rows before treating the simulator as a final rules-perfect engine.")
    st.markdown("#### Persistence")
    st.write("The model is saved to a local JSON file when the host permits it, and can be downloaded here. Streamlit Community Cloud may reset local files during restarts or redeployments, so download the JSON checkpoint after training and re-upload/persist it if needed.")