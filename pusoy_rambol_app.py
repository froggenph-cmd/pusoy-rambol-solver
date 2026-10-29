import streamlit as st
from itertools import combinations
from collections import Counter
import random, json, os, math

st.set_page_config(page_title="Pusoy Rambol Solver & Learner", page_icon="🃏", layout="wide")
st.title("🃏 Pusoy Rambol Solver & Self-Play Learner")
st.caption("Player 1 strategy learner · A♦ randomly dealt · Special hands checked first")

RANKS = "23456789TJQKA"
SUITS = "CDHS"
RV = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_SYMBOL = {"C":"♣", "D":"♦", "H":"♥", "S":"♠"}
RANK_LABEL = {**{str(i):str(i) for i in range(2,10)}, "T":"10", "J":"J", "Q":"Q", "K":"K", "A":"A"}
DECK = [r+s for r in RANKS for s in SUITS]
MODEL_PATH = "pusoy_rambol_model.json"
DEFAULT_WEIGHTS = {"bottom":1.0,"middle":1.0,"top":1.0,"top_kicker":0.08,"balance":0.18,"category":0.02}


def card_label(c):
    # Emoji presentation keeps hearts/diamonds visibly red on most mobile devices.
    suit = SUIT_SYMBOL[c[1]] + ("︎" if c[1] in ("C", "S") else "️")
    return f"{RANK_LABEL[c[0]]}{suit}"

def show_cards(cards): return " ".join(card_label(c) for c in cards)

# Give interactive card buttons a playing-card face rather than the default flat button look.
st.markdown(""" <style> /* Card-picker buttons: white face, rounded corners, subtle border and shadow. */ div[data-testid="stButton"] > button { min-height: 48px; padding: 0.35rem 0.2rem; border: 1px solid #b9bec7; border-radius: 9px; background: linear-gradient(145deg, #ffffff 0%, #f7f7f7 100%); color: #20242b; box-shadow: 0 2px 3px rgba(0,0,0,.14); font-size: 1.05rem; font-weight: 650; } div[data-testid="stButton"] > button:hover { border-color: #687385; background: #fff; transform: translateY(-1px); } div[data-testid="stButton"] > button[kind="primary"] { border: 2px solid #3977c3; background: #e8f1ff; color: #173e72; } </style> """, unsafe_allow_html=True)
def straight_high(values):
    vals = sorted(set(values))
    if len(vals) != 5: return None
    if vals == [2,3,4,5,14]: return 5
    return vals[-1] if vals[-1] - vals[0] == 4 else None

def detect_special_hand(hand):
    """Return all qualifying special-hand labels found anywhere in a 13-card hand. A special hand qualifies if it contains a five-card straight flush, four of a kind, or six pair-units. A four-of-a-kind contributes two pair-units toward six pairs. """
    labels = []
    rank_values = [RV[c[0]] for c in hand]
    rank_counts = Counter(rank_values)

    # Four of a kind (or better, if a malformed input were ever passed).
    four_kind = any(count >= 4 for count in rank_counts.values())
    if four_kind:
        labels.append("Four of a kind")

    # Count pair-units, not just distinct ranks that appear at least twice.
    # Example: four Kings count as two pairs toward the six-pairs requirement.
    pair_units = sum(count // 2 for count in rank_counts.values())
    if pair_units >= 6:
        labels.append("Six pairs")

    # Find any five-card straight flush among all 13 cards, including A-2-3-4-5.
    for suit in SUITS:
        suited_values = sorted({RV[c[0]] for c in hand if c[1] == suit})
        if len(suited_values) < 5:
            continue
        # Ace can be low for A-2-3-4-5 as well as high for 10-J-Q-K-A.
        if 14 in suited_values:
            suited_values = sorted(set(suited_values + [1]))
        for start in range(len(suited_values) - 4):
            window = suited_values[start:start + 5]
            if window[-1] - window[0] == 4:
                labels.append("Straight flush")
                return labels
    return labels


def rank_five(cards):
    vals = [RV[c[0]] for c in cards]; counts = Counter(vals)
    groups = sorted(((n,v) for v,n in counts.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1; sh = straight_high(vals)
    if flush and sh: return (8, sh)
    if groups[0][0] == 4:
        q = groups[0][1]; return (7,q,max(v for v in vals if v != q))
    if sorted(counts.values()) == [2,3]:
        return (6,max(v for v,n in counts.items() if n == 3),max(v for v,n in counts.items() if n == 2))
    if flush: return (5,*sorted(vals,reverse=True))
    if sh: return (4,sh)
    if groups[0][0] == 3:
        t=groups[0][1]; return (3,t,*sorted((v for v in vals if v != t),reverse=True))
    pairs=sorted((v for v,n in counts.items() if n == 2),reverse=True)
    if len(pairs)==2: return (2,pairs[0],pairs[1],max(v for v in vals if v not in pairs))
    if len(pairs)==1:
        p=pairs[0]; return (1,p,*sorted((v for v in vals if v != p),reverse=True))
    return (0,*sorted(vals,reverse=True))

def rank_top(cards):
    vals=[RV[c[0]] for c in cards]; counts=Counter(vals)
    groups=sorted(((n,v) for v,n in counts.items()),reverse=True)
    if groups[0][0]==3: return (3,groups[0][1])
    if groups[0][0]==2:
        p=groups[0][1]; return (1,p,max(v for v in vals if v != p))
    return (0,*sorted(vals,reverse=True))

def category(rank, top=False):
    names=["High card","One pair","Two pair","Three of a kind","Straight","Flush","Full house","Four of a kind","Straight flush"]
    return ({0:"High card",1:"One pair",3:"Three of a kind"}.get(rank[0],"High card") if top else names[rank[0]])

def foul_free(bottom,middle,top):
    rb,rm,rt=rank_five(bottom),rank_five(middle),rank_top(top)
    # Retains the app's current rule; confirm cross-row rules with the user before treating as final.
    return rb >= rm and rm[0] >= rt[0]

def tie_value(rank):
    value=rank[0]*100000.0
    for i,x in enumerate(rank[1:]): value += x*(1000.0/(15**i))
    return value

def arrangement_candidates(hand, sample_count=450, rng=None):
    rng=rng or random; hand=tuple(hand); found={}
    if sample_count is None:
        for b in combinations(hand,5):
            bset=set(b); rest=tuple(c for c in hand if c not in bset); rb=rank_five(b)
            for m in combinations(rest,5):
                mset=set(m); top=tuple(c for c in rest if c not in mset)
                rm,rt=rank_five(m),rank_top(top)
                if rb < rm or rm[0] < rt[0]: continue
                key=(tuple(sorted(b)),tuple(sorted(m)),tuple(sorted(top)))
                found[key]=(b,m,top,rb,rm,rt)
    else:
        for _ in range(sample_count):
            perm=list(hand); rng.shuffle(perm); b,m,top=tuple(perm[:5]),tuple(perm[5:10]),tuple(perm[10:13])
            rb,rm,rt=rank_five(b),rank_five(m),rank_top(top)
            if rb < rm or rm[0] < rt[0]: continue
            key=(tuple(sorted(b)),tuple(sorted(m)),tuple(sorted(top)))
            found[key]=(b,m,top,rb,rm,rt)
        # Add high-value bottom/middle combinations for better sample coverage.
        for b in combinations(hand,5):
            rb=rank_five(b)
            if rb[0] < 2: continue
            rest=tuple(c for c in hand if c not in set(b))
            for m in combinations(rest,5):
                rm=rank_five(m)
                if rb < rm: continue
                top=tuple(c for c in rest if c not in set(m)); rt=rank_top(top)
                if rm[0] < rt[0]: continue
                key=(tuple(sorted(b)),tuple(sorted(m)),tuple(sorted(top)))
                found.setdefault(key,(b,m,top,rb,rm,rt))
                if len(found)>=sample_count+180: break
            if len(found)>=sample_count+180: break
    return list(found.values())

def strategy_score(item,weights):
    b,m,t,rb,rm,rt=item
    vals=[tie_value(rb)/1000.0,tie_value(rm)/1000.0,tie_value(rt)/1000.0]
    score=weights["bottom"]*vals[0]+weights["middle"]*vals[1]+weights["top"]*vals[2]
    score += weights["category"]*(rb[0]+rm[0]+rt[0]) + weights["balance"]*min(vals)
    score += weights["top_kicker"]*sum(sorted((RV[c[0]] for c in t),reverse=True))
    return score

def choose_arrangement(hand,weights,mode="learned",rng=None):
    rng=rng or random; candidates=arrangement_candidates(hand,450,rng)
    if not candidates: candidates=arrangement_candidates(hand,None,rng)
    if not candidates: return None
    if mode=="random": return rng.choice(candidates)
    if mode=="strong": w={"bottom":1.0,"middle":1.0,"top":1.0,"top_kicker":0.01,"balance":0.25,"category":0.04}
    elif mode=="balanced": w={"bottom":0.85,"middle":1.2,"top":1.0,"top_kicker":0.03,"balance":0.4,"category":0.03}
    else: w=weights
    return max(candidates,key=lambda x:strategy_score(x,w))

def head_to_head(a,b,hand_a,hand_b):
    wins_a=wins_b=0
    for ra,rb in zip(a[3:],b[3:]):
        cmp=(ra>rb)-(ra<rb)
        if cmp==0 and ("AD" in hand_a) != ("AD" in hand_b): cmp=1 if "AD" in hand_a else -1
        if cmp>0: wins_a+=1
        elif cmp<0: wins_b+=1
    return 1 if wins_a>=2 else (-1 if wins_b>=2 else 0)

def deal_game(rng):
    deck=DECK[:]; rng.shuffle(deck); hands=[deck[i*13:(i+1)*13] for i in range(4)]
    assert all(len(h)==13 and len(set(h))==13 for h in hands) and sum("AD" in h for h in hands)==1
    return hands

def play_deal(hands,weights,deal_seed,opponent_modes=("random","strong","balanced")):
    # Special hands are checked across all 13 cards before row-by-row comparison.
    special_hands = [detect_special_hand(hand) for hand in hands]
    qualifies = [bool(labels) for labels in special_hands]

    # Independent but repeatable arrangement RNGs make candidate policies face the same deals
    # and the same random-opponent sampling for a fair paired comparison.
    arrangements=[]
    for i,hand in enumerate(hands):
        rng=random.Random(deal_seed*1009+i*9176+31)
        mode="learned" if i==0 else opponent_modes[i-1]
        arrangement = choose_arrangement(hand,weights if i==0 else DEFAULT_WEIGHTS,mode,rng)
        if arrangement is None:
            raise RuntimeError("Could not find a foul-free arrangement for a dealt hand.")
        arrangements.append(arrangement)

    results=[]
    for j in range(1,4):
        if qualifies[0] and qualifies[j]:
            # Two qualifying special hands tie against each other.
            result=0
        elif qualifies[0]:
            result=1  # A special hand beats a non-special hand 3-0.
        elif qualifies[j]:
            result=-1 # Opponent's special hand beats Player 1 3-0.
        else:
            result=head_to_head(arrangements[0],arrangements[j],hands[0],hands[j])
        results.append(result)

    holder=next(i for i,h in enumerate(hands) if "AD" in h)
    return {
        "results":results,
        "p1_has_ad":holder==0,
        "ad_holder":holder,
        "vs_ad_holder":None if holder==0 else results[holder-1],
        "special_hands":special_hands,
        "p1_special_hand":special_hands[0],
    }


def compare_arrangements(arr_a, arr_b, hand_a, hand_b, special_a=None, special_b=None):
    """Compare two arrangements while applying whole-hand special rules first."""
    special_a = detect_special_hand(hand_a) if special_a is None else special_a
    special_b = detect_special_hand(hand_b) if special_b is None else special_b
    if special_a and special_b:
        return 0
    if special_a:
        return 1
    if special_b:
        return -1
    return head_to_head(arr_a, arr_b, hand_a, hand_b)


def learn_from_human_game(model, human_hand, human_arrangement, opponent_hands, seed):
    """Try small weight mutations and keep one only if it performs better on this game."""
    rng = random.Random(seed)
    current = normalize_weights(model["weights"])
    human_special = detect_special_hand(human_hand)

    def score_weights(candidate):
        outcomes = []
        for i, opp_hand in enumerate(opponent_hands):
            rr = random.Random(seed * 1009 + i * 9176 + 31)
            opp_arr = choose_arrangement(opp_hand, candidate, "learned", rr)
            if opp_arr is None:
                outcomes.append(-1)
            else:
                outcomes.append(compare_arrangements(opp_arr, human_arrangement, opp_hand, human_hand))
        # Positive means learner-controlled opponents beat the human.
        return sum(outcomes), outcomes

    before_score, before_outcomes = score_weights(current)
    best_weights = current
    best_score = before_score
    for _ in range(5):
        candidate = current.copy()
        for key in candidate:
            if rng.random() < 0.7:
                candidate[key] = max(0.001, min(3.0, candidate[key] * rng.uniform(0.75, 1.30)))
        candidate_score, _ = score_weights(candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_weights = candidate
    accepted = best_score > before_score
    if accepted:
        model["weights"] = normalize_weights(best_weights)
    model["human_games"] = int(model.get("human_games", 0)) + 1
    model.setdefault("human_history", []).append({
        "game": model["human_games"],
        "before_opponent_score": before_score,
        "after_opponent_score": best_score if accepted else before_score,
        "opponent_results": before_outcomes,
        "weight_update_accepted": accepted,
        "human_special_hand": human_special,
    })
    model["human_history"] = model["human_history"][-100:]
    return model, before_outcomes, accepted

def initial_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH,"r",encoding="utf-8") as f: data=json.load(f)
            if isinstance(data.get("weights"),dict): return data
        except Exception: pass
    return {"weights":DEFAULT_WEIGHTS.copy(),"games":0,"history":[],"version":2}

def normalize_weights(w):
    return {k:max(0.001,min(3.0,float(w.get(k,DEFAULT_WEIGHTS[k])))) for k in DEFAULT_WEIGHTS}

def make_deals(n,seed):
    rng=random.Random(seed)
    return [(deal_game(rng),rng.randrange(1,2**31-1)) for _ in range(n)]

def evaluate(weights,deals):
    ad_games=ad_wins=0; no_ad_games=no_ad_wins=0; p1_match_wins=p1_draws=0; opponent_wins=0
    for hands,seed in deals:
        r=play_deal(hands,weights,seed)
        w=sum(x==1 for x in r["results"]); d=sum(x==0 for x in r["results"])
        p1_match_wins += w; p1_draws += d
        opponent_wins += sum(x==-1 for x in r["results"])
        if r["p1_has_ad"]:
            ad_games+=1; ad_wins+=w
        else:
            no_ad_games+=1
            if r["vs_ad_holder"]==1: no_ad_wins+=1
    ad_rate=ad_wins/(3*ad_games) if ad_games else 0.0
    vs_rate=no_ad_wins/no_ad_games if no_ad_games else 0.0
    overall=p1_match_wins/(3*len(deals)) if deals else 0.0
    score=0.45*ad_rate+0.45*vs_rate+0.10*overall
    return {"score":score,"ad_games":ad_games,"ad_rate":ad_rate,"no_ad_games":no_ad_games,"vs_ad_rate":vs_rate,"avg_wins":p1_match_wins/len(deals) if deals else 0.0,"overall_match_win_rate":overall,"draws":p1_draws,"opponent_wins":opponent_wins,"games":len(deals)}

def train(model,games,seed):
    rng=random.Random(seed); current=normalize_weights(model["weights"])
    # Training and held-out evaluation use separate deals. All candidate policies see identical
    # training deals; a candidate is accepted only if held-out score improves over the incumbent.
    train_n=max(20,games); test_n=max(50,min(200,games))
    train_deals=make_deals(train_n,seed+100)
    test_deals=make_deals(test_n,seed+900000)
    candidates=[current.copy()]
    for _ in range(4):
        mutant=current.copy()
        for k in mutant:
            if rng.random()<0.65: mutant[k]=max(0.001,min(3.0,mutant[k]*rng.uniform(0.72,1.32)))
        candidates.append(mutant)
    train_reports=[evaluate(w,train_deals) for w in candidates]
    # Pick on train data, then compare against incumbent on held-out test deals.
    proposed=candidates[max(range(len(candidates)),key=lambda i:train_reports[i]["score"]) ]
    before=evaluate(current,test_deals); proposed_test=evaluate(proposed,test_deals)
    accepted=proposed_test["score"] > before["score"]
    chosen=proposed if accepted else current
    after=proposed_test if accepted else before
    model["weights"]=normalize_weights(chosen)
    model["games"]=int(model.get("games",0))+train_n
    hist=model.setdefault("history",[])
    hist.append({"total_games":model["games"],"batch_games":train_n,"accepted":accepted,"train_score":round(max(x["score"] for x in train_reports),4),"test_before":round(before["score"],4),"test_after":round(after["score"],4),"ad_rate":round(after["ad_rate"],4),"vs_ad_rate":round(after["vs_ad_rate"],4),"avg_wins":round(after["avg_wins"],3),"test_games":test_n})
    model["history"]=hist[-100:]
    return model,{"accepted":accepted,"before":before,"after":after,"candidate_train_best":max(x["score"] for x in train_reports),"training_games":train_n,"test_games":test_n}

def save_model(model):
    with open(MODEL_PATH,"w",encoding="utf-8") as f: json.dump(model,f,indent=2)

model=initial_model()
if "model_state" not in st.session_state: st.session_state.model_state=model
model=st.session_state.model_state
weights=normalize_weights(model["weights"])
solver_tab,learner_tab,play_tab,rules_tab=st.tabs(["Hand solver","Self-play learner","Play vs learner","Rules & model"])

with solver_tab:
    st.subheader("Find strong arrangements for your 13 cards")
    st.info("Tap the playing cards to select or unselect them. Hearts and diamonds are red; clubs and spades are black. Select exactly 13, then find strong arrangements.")
    chosen_key = "solver_selected_cards"
    if chosen_key not in st.session_state:
        st.session_state[chosen_key] = []
    selected_cards = st.session_state[chosen_key]
    st.write(f"**Selected: {len(selected_cards)}/13 cards**")
    if st.button("Clear selected cards", key="solver_clear_cards"):
        st.session_state[chosen_key] = []
        st.session_state.pop("solver_results", None)
        st.rerun()
    for row_start in range(0, 52, 4):
        cols = st.columns(4)
        for j, card in enumerate(DECK[row_start:row_start+4]):
            with cols[j]:
                chosen = card in selected_cards
                label = ("✓ " if chosen else "") + card_label(card)
                if st.button(label, key=f"solver_card_{card}", use_container_width=True, type="primary" if chosen else "secondary"):
                    current = list(st.session_state[chosen_key])
                    if card in current:
                        current.remove(card)
                    elif len(current) < 13:
                        current.append(card)
                    st.session_state[chosen_key] = current
                    st.rerun()
    if st.button("Find strong arrangements", type="primary", use_container_width=True, disabled=len(selected_cards) != 13, key="solver_find_arrangements"):
        cards = list(selected_cards)
        with st.spinner("Searching legal arrangements..."):
            results = arrangement_candidates(cards, sample_count=None)
            results.sort(key=lambda x:strategy_score(x,weights), reverse=True)
            unique=[]; seen=set()
            for item in results:
                sig=(item[3],item[4],item[5])
                if sig in seen: continue
                seen.add(sig); unique.append(item)
                if len(unique)>=10: break
            st.session_state.solver_results = (len(results), unique)
    if len(selected_cards) == 13 and st.session_state.get("solver_results"):
        total, unique = st.session_state.solver_results
        st.write(f"Found **{total:,} legal arrangements**. Showing up to 10 distinct hand-strength profiles.")
        for idx,item in enumerate(unique,1):
            b,m,t,rb,rm,rt=item
            with st.container(border=True):
                st.subheader(f"Option {idx}"); c1,c2,c3=st.columns(3)
                with c1: st.markdown("**BOTTOM · 5**"); st.write(show_cards(b)); st.caption(category(rb))
                with c2: st.markdown("**MIDDLE · 5**"); st.write(show_cards(m)); st.caption(category(rm))
                with c3: st.markdown("**TOP · 3**"); st.write(show_cards(t)); st.caption(category(rt,True))
                st.caption(f"Policy score: {strategy_score(item,weights):,.1f}")

with learner_tab:
    st.subheader("Player 1 controlled self-play learning")
    st.write("A♦ is shuffled into one of the four hands. Player 1 learns to maximize wins against all three opponents when holding A♦, and prioritize beating the A♦ holder otherwise.")
    c1,c2,c3=st.columns(3)
    c1.metric("Training deals recorded",f"{model.get('games',0):,}")
    last=(model.get("history") or [{}])[-1]
    c2.metric("Test score after",f"{last.get('test_after',0)*100:.1f}%" if model.get("history") else "Not measured")
    c3.metric("Latest change", "Accepted" if last.get("accepted") else ("Rejected" if model.get("history") else "Not measured"))
    st.info("Fairness fix: every candidate sees the same training deals. A proposed change is kept only if its score improves on a separate held-out test set. This is still a heuristic evolutionary learner, not a proven optimal solver.")
    n_games=st.select_slider("Training deals per batch",options=[25,50,100,250,500],value=50,help="The app uses at least 20 training deals and a separate held-out test set of 50–200 deals. Evaluation can take a few minutes.")
    seed=st.number_input("Random seed",min_value=0,max_value=2_147_483_647,value=20261028,step=1)
    if st.button("Train and test learner",type="primary",use_container_width=True):
        with st.spinner("Evaluating candidate strategies on matched deals, then testing on unseen deals..."):
            try:
                new_model,report=train({"weights":weights,"games":model.get("games",0),"history":model.get("history",[]),"version":2},int(n_games),int(seed))
                st.session_state.model_state=new_model
                try: save_model(new_model)
                except Exception as e: st.warning(f"Local checkpoint could not be saved by the host: {e}. Download the JSON checkpoint below.")
                st.success("Training and independent test completed.")
                st.write("Strategy change:","Accepted because held-out score improved." if report["accepted"] else "Rejected because held-out score did not improve.")
                a,b=st.columns(2)
                with a: st.metric("Held-out score before",f"{report['before']['score']*100:.2f}%")
                with b: st.metric("Held-out score after",f"{report['after']['score']*100:.2f}%",delta=f"{(report['after']['score']-report['before']['score'])*100:+.2f} percentage points")
                st.write(f"Held-out deals: {report['test_games']}; training deals: {report['training_games']}.")
                st.rerun()
            except Exception as e: st.error(f"Training stopped: {e}")
    if model.get("history"):
        st.markdown("#### Recent training history")
        st.dataframe([{"Total training deals":h["total_games"],"Batch deals":h["batch_games"],"Change accepted":h.get("accepted",False),"Test score before":f"{h.get('test_before',0)*100:.2f}%","Test score after":f"{h.get('test_after',0)*100:.2f}%","A♦ holder win fraction":f"{h['ad_rate']*100:.1f}%","Win vs A♦ holder":f"{h['vs_ad_rate']*100:.1f}%","Average H2H wins/game":h["avg_wins"],"Test deals":h.get("test_games",0)} for h in model["history"][-10:]],use_container_width=True,hide_index=True)
    st.markdown("#### Current learned weights"); st.json(weights)
    st.download_button("Download learned model JSON",data=json.dumps(model,indent=2),file_name="pusoy_rambol_model.json",mime="application/json",use_container_width=True)
    uploaded=st.file_uploader("Restore a previously downloaded model checkpoint",type=["json"])
    if uploaded is not None and st.button("Load checkpoint"):
        try:
            loaded=json.load(uploaded)
            if not isinstance(loaded.get("weights"),dict): raise ValueError("This JSON does not contain model weights.")
            loaded["weights"]=normalize_weights(loaded["weights"]); st.session_state.model_state=loaded
            st.success("Checkpoint loaded into this app session."); st.rerun()
        except Exception as e: st.error(f"Could not load checkpoint: {e}")
    if st.button("Reset learner to default",type="secondary"):
        st.session_state.model_state={"weights":DEFAULT_WEIGHTS.copy(),"games":0,"history":[],"version":2}
        try:
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        except Exception: pass
        st.rerun()


with play_tab:
    st.subheader("Play a real deal against the learner")
    st.write("Choose whether to receive a random 13-card hand or select the exact 13 cards you want. The three learner opponents receive the remaining cards from the same 52-card deck, so no card can appear twice.")
    st.info("This is a lightweight online-learning step, not a guarantee of long-term improvement: one deal is noisy, and accepted updates are based on the current game. Your learned weights are saved to the same model checkpoint.")

    hand_mode = st.radio(
        "How would you like your hand?",
        ["Deal random cards", "Select my own cards"],
        horizontal=True,
        key="human_hand_mode",
    )

    def reset_arrangement_state():
        for key in ("human_bottom", "human_middle", "human_top", "human_bottom_input", "human_middle_input", "human_top_input"):
            st.session_state.pop(key, None)
        st.session_state.pop("human_game_result", None)

    if hand_mode == "Deal random cards":
        st.caption("A fresh shuffled deck is used when you deal. Your opponents receive the other 39 cards. A♦ breaks an otherwise tied row comparison when exactly one player holds it.")
        if st.button("Deal me 13 random cards", type="primary", use_container_width=True, key="deal_random_human_hand"):
            shuffled_deck = DECK.copy()
            random.SystemRandom().shuffle(shuffled_deck)
            st.session_state.human_hand = shuffled_deck[:13]
            st.session_state.human_predealt_opponents = [shuffled_deck[13+i*13:13+(i+1)*13] for i in range(3)]
            reset_arrangement_state()
            st.session_state.human_rows = {"Bottom": [], "Middle": [], "Top": []}
            st.session_state.human_selected_for_move = None
            st.rerun()
    else:
        st.markdown("**Tap 13 cards to select your hand**")
        st.caption("Hearts and diamonds are red; clubs and spades are black. A♦ is dealt normally and its tie-break rule applies during play.")
        select_key = "human_custom_selected_cards"
        if select_key not in st.session_state:
            st.session_state[select_key] = []
        chosen_custom = st.session_state[select_key]
        st.write(f"**Selected: {len(chosen_custom)}/13 cards**")
        if st.button("Clear card selection", key="human_custom_clear"):
            st.session_state[select_key] = []
            st.rerun()
        for row_start in range(0, 52, 4):
            cols = st.columns(4)
            for j, card in enumerate(DECK[row_start:row_start+4]):
                with cols[j]:
                    is_chosen = card in chosen_custom
                    if st.button(("✓ " if is_chosen else "") + card_label(card), key=f"human_custom_card_{card}", use_container_width=True, type="primary" if is_chosen else "secondary"):
                        current = list(st.session_state[select_key])
                        if card in current: current.remove(card)
                        elif len(current) < 13: current.append(card)
                        st.session_state[select_key] = current
                        st.rerun()
        if st.button("Confirm these 13 cards", use_container_width=True, disabled=len(chosen_custom) != 13, key="human_custom_confirm"):
            st.session_state.human_hand = list(chosen_custom)
            st.session_state.pop("human_predealt_opponents", None)
            reset_arrangement_state()
            st.session_state.human_rows = {"Bottom": [], "Middle": [], "Top": []}
            st.session_state.human_selected_for_move = None
            st.rerun()

    human_hand = st.session_state.get("human_hand")
    if human_hand:
        st.markdown("### Your 13 cards")
        st.caption("A♦ rule: if a row comparison is otherwise tied, the player holding A♦ anywhere in their 13-card hand wins that row. Special-hand rules take priority.")
        st.caption("Tap a card to select it, then tap the row where you want to place it. Tap a card already in a row to select it and move it to another row.")
        rows = st.session_state.setdefault("human_rows", {"Bottom": [], "Middle": [], "Top": []})
        # Reconcile saved row state with the current hand.
        if set(sum(rows.values(), [])) - set(human_hand):
            rows = {"Bottom": [], "Middle": [], "Top": []}
            st.session_state.human_rows = rows
        assigned = sum(rows.values(), [])
        unassigned = [c for c in human_hand if c not in assigned]
        selected_move = st.session_state.get("human_selected_for_move")
        st.markdown("**Cards not yet placed**")
        if not unassigned:
            st.caption("All cards have been placed in rows.")
        for row_start in range(0, len(unassigned), 7):
            cols = st.columns(7)
            for j, card in enumerate(unassigned[row_start:row_start+7]):
                with cols[j]:
                    if st.button(("👉 " if selected_move == card else "") + card_label(card), key=f"human_unassigned_{card}", use_container_width=True, type="primary" if selected_move == card else "secondary"):
                        st.session_state.human_selected_for_move = None if selected_move == card else card
                        st.rerun()
        row_cols = st.columns(3)
        limits = {"Bottom":5, "Middle":5, "Top":3}
        for idx, row_name in enumerate(("Bottom", "Middle", "Top")):
            with row_cols[idx]:
                st.markdown(f"**{row_name} · {limits[row_name]} cards**")
                st.write(show_cards(rows[row_name]) if rows[row_name] else "— empty —")
                for card in list(rows[row_name]):
                    if st.button(("👉 " if selected_move == card else "") + card_label(card), key=f"human_row_card_{row_name}_{card}", use_container_width=True, type="primary" if selected_move == card else "secondary"):
                        st.session_state.human_selected_for_move = None if selected_move == card else card
                        st.rerun()
                if st.button(f"Place selected here ({len(rows[row_name])}/{limits[row_name]})", key=f"human_place_{row_name}", use_container_width=True, disabled=selected_move is None or len(rows[row_name]) >= limits[row_name]):
                    for old_row in rows.values():
                        if selected_move in old_row: old_row.remove(selected_move)
                    rows[row_name].append(selected_move)
                    st.session_state.human_rows = rows
                    st.session_state.human_selected_for_move = None
                    st.rerun()
        if st.button("Return selected card to unassigned", key="human_unassign_selected", use_container_width=True, disabled=selected_move is None):
            for old_row in rows.values():
                if selected_move in old_row: old_row.remove(selected_move)
            st.session_state.human_rows = rows
            st.session_state.human_selected_for_move = None
            st.rerun()
        st.write(f"**Placed:** {len(assigned)}/13 cards")
        ready = len(rows["Bottom"]) == 5 and len(rows["Middle"]) == 5 and len(rows["Top"]) == 3 and len(assigned) == 13
        if st.button("Play this arrangement", type="primary", use_container_width=True, disabled=not ready, key="human_play_arrangement"):
            b, m, t = list(rows["Bottom"]), list(rows["Middle"]), list(rows["Top"])
            human_arr = (tuple(b), tuple(m), tuple(t), rank_five(b), rank_five(m), rank_top(t))
            if not foul_free(b, m, t):
                st.error("This arrangement is foul under the app's current rule: Bottom must be at least as strong as Middle, and Middle category must be at least Top category.")
            else:
                opponent_hands = st.session_state.get("human_predealt_opponents")
                if not opponent_hands or len(opponent_hands) != 3 or set(sum(opponent_hands, [])) != set(DECK) - set(human_hand):
                    remaining = [c for c in DECK if c not in set(human_hand)]
                    rr = random.Random(random.SystemRandom().randrange(1, 2**31-1)); rr.shuffle(remaining)
                    opponent_hands = [remaining[i*13:(i+1)*13] for i in range(3)]
                game_seed = random.SystemRandom().randrange(1, 2**31-1)
                opponent_arrangements = []
                for i, opp_hand in enumerate(opponent_hands):
                    opp_rng = random.Random(game_seed * 1009 + i * 9176 + 31)
                    opponent_arrangements.append(choose_arrangement(opp_hand, normalize_weights(st.session_state.model_state["weights"]), "learned", opp_rng))
                human_special = detect_special_hand(human_hand)
                # Keep human-vs-each-opponent results for learner feedback, but also
                # compare every other player directly against the A♦ holder.
                results = [compare_arrangements(human_arr, opp_arr, human_hand, opp_hand) for opp_arr, opp_hand in zip(opponent_arrangements, opponent_hands)]
                all_hands = [human_hand] + opponent_hands
                all_arrangements = [human_arr] + opponent_arrangements
                ad_holder_index = next((i for i, hand in enumerate(all_hands) if "AD" in hand), None)
                ad_comparisons = []
                if ad_holder_index is not None:
                    for i in range(4):
                        if i == ad_holder_index:
                            continue
                        # Outcome is from the A♦ holder's perspective: 1=holder wins, -1=other player wins.
                        ad_comparisons.append({
                            "player_index": i,
                            "outcome": compare_arrangements(
                                all_arrangements[ad_holder_index], all_arrangements[i],
                                all_hands[ad_holder_index], all_hands[i]
                            )
                        })
                st.session_state.human_game_result = {
                    "human_hand": human_hand, "human_arrangement": human_arr,
                    "opponent_hands": opponent_hands, "opponent_arrangements": opponent_arrangements,
                    "results": results, "human_special": human_special, "seed": game_seed,
                    "ad_holder_index": ad_holder_index, "ad_comparisons": ad_comparisons
                }
                st.rerun()

    game = st.session_state.get("human_game_result")
    if game:
        st.markdown("### Deal result")
        all_hands = [game["human_hand"]] + game["opponent_hands"]
        all_arrangements = [game["human_arrangement"]] + game["opponent_arrangements"]
        ad_holder_index = game.get("ad_holder_index", next((i for i, hand in enumerate(all_hands) if "AD" in hand), 0))
        ad_holder_name = "You" if ad_holder_index == 0 else f"Opponent {ad_holder_index}"
        st.info(f"**A♦ holder: {ad_holder_name}.** The other three players are compared directly against this player's arrangement. A♦ breaks otherwise-tied rows in the holder's favor; special-hand rules take priority.")
        st.markdown("### Results against the A♦ holder")
        for comparison in game.get("ad_comparisons", []):
            player_index = comparison["player_index"]
            outcome = comparison["outcome"]
            other_name = "You" if player_index == 0 else f"Opponent {player_index}"
            if outcome == 1:
                result_text = f"{ad_holder_name} wins 2 or more rows"
            elif outcome == -1:
                result_text = f"{other_name} wins 2 or more rows"
            else:
                result_text = "Head-to-head draw"
            st.write(f"- **{other_name} vs {ad_holder_name}:** {result_text}")
        if game["human_special"]:
            st.success("Your special hand: " + ", ".join(game["human_special"]))
        if game["human_special"]:
            st.success("Your special hand: " + ", ".join(game["human_special"]))
        hcols = st.columns(3)
        for i, (opp_hand, opp_arr, outcome) in enumerate(zip(game["opponent_hands"], game["opponent_arrangements"], game["results"])):
            label = "Learner wins" if outcome == 1 else ("You win" if outcome == -1 else "Tie")
            with hcols[i]:
                st.markdown(f"**Opponent {i+1} — {label}**")
                if detect_special_hand(opp_hand):
                    st.caption("Special: " + ", ".join(detect_special_hand(opp_hand)))
                st.write("**Bottom:** " + show_cards(opp_arr[0]))
                st.caption(category(opp_arr[3]))
                st.write("**Middle:** " + show_cards(opp_arr[1]))
                st.caption(category(opp_arr[4]))
                st.write("**Top:** " + show_cards(opp_arr[2]))
                st.caption(category(opp_arr[5], True))
        human_wins = sum(x == -1 for x in game["results"])
        learner_wins = sum(x == 1 for x in game["results"])
        ties = sum(x == 0 for x in game["results"])
        st.write(f"**Your head-to-head record:** {human_wins} wins · {ties} ties · {learner_wins} learner wins")
        if st.button("Record result and let learner learn", type="primary", use_container_width=True, key="learn_from_human_game"):
            current_model = st.session_state.model_state
            new_model, old_results, accepted = learn_from_human_game(
                current_model, game["human_hand"], game["human_arrangement"], game["opponent_hands"], game["seed"]
            )
            st.session_state.model_state = new_model
            try:
                save_model(new_model)
            except Exception as e:
                st.warning(f"The host could not save a local checkpoint: {e}. Download the model JSON below.")
            st.session_state.human_game_result = None
            st.success("Game recorded. " + ("A weight update was accepted." if accepted else "No candidate weight update improved this deal, so the weights stayed the same."))
            st.rerun()
        st.caption(f"Human games recorded: {model.get('human_games', 0)}")

with rules_tab:
    st.subheader("Rules currently encoded")
    st.markdown("- Play vs learner lets you arrange a hand against three learner-controlled opponents and record results for online weight updates.\n- Standard 52-card deck; four players receive 13 cards each.\n- A♦ is randomly distributed by shuffling and dealing.\n- Bottom 5, Middle 5, Top 3; no card reused.\n- Five-card ranking: Straight Flush > Four of a Kind > Full House > Flush > Straight > Three of a Kind > Two Pair > One Pair > High Card.\n- Top ranking: Three of a Kind > Pair > High Card; straights and flushes do not count in Top.\n- Special hands are checked across all 13 cards before ordinary row comparisons: any five-card straight flush in one suit, four of a kind, or six pair-units.\n- Four of a kind counts as two pairs toward the six-pairs special hand.\n- A qualifying special hand beats every non-qualifying hand 3–0 regardless of arrangement. If both players qualify, they tie against each other.\n- Ordinary head-to-head win: win at least two of the three corresponding rows.\n- A♦ breaks otherwise-equal row ties if exactly one player holds it anywhere in their hand.\n- Current foul check: Bottom rank ≥ Middle rank and Middle category ≥ Top category.")
    st.warning("The cross-row foul rule remains simplified from the existing app and must be confirmed against your table's exact rule before treating the simulator as rules-perfect.")
    st.write("Download the JSON checkpoint after training. Streamlit Community Cloud may reset local files on restart or redeployment.")