# app_dataset.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random, time, datetime, json, re, csv
from pathlib import Path
from rag_engine import RAGTutor

st.set_page_config(page_title="🎮 Dynamic Game-Based Tutor", layout="centered")

DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
SUBMIT_DIR = DATA_DIR / "submissions"
for d in [DATA_DIR, PROFILES_DIR, SUBMIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_rag():
    return RAGTutor(str(DATA_DIR / "final_dataset.csv"))

rag_tutor = load_rag()
# -------------------- LOAD QUESTION BANK --------------------
@st.cache_data
def load_bank():
    df = pd.read_csv(DATA_DIR / "final_dataset.csv")
    # expected columns: id, subject, topic, difficulty, question, options, correct_option, explanation
    df["options_list"] = df["options"].astype(str).apply(lambda s: s.split("|"))
    valid = {"Easy", "Medium", "Hard"}
    if not set(df["difficulty"].unique()).issubset(valid):
        def _len2diff(q):
            L = len(str(q))
            return "Easy" if L < 80 else ("Medium" if L < 160 else "Hard")
        df["difficulty"] = df.apply(
            lambda r: r["difficulty"] if r["difficulty"] in valid else _len2diff(r["question"]),
            axis=1
        )
    if "explanation" not in df.columns:
        df["explanation"] = ""
    else:
        df["explanation"] = df["explanation"].astype(str).replace({"nan": ""})
    df["id"] = df["id"].astype(str)
    return df

try:
    BANK = load_bank()
except Exception as e:
    st.error("⚠️ data/question_bank.csv not found or unreadable. Run your prep script to create it.")
    st.stop()

# -------------------- UTILITIES --------------------
DIFFS = ["Easy", "Medium", "Hard"]

def slug(name: str) -> str:
    return "_".join(name.strip().lower().split())

def profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{slug(name)}.json"

def load_profile(name: str) -> dict:
    p = profile_path(name)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    now = time.time()
    return {
        "name": name,
        "created_at": now,
        "last_seen": now,
        "cumulative": {
            "total": 0,
            "correct": 0,
            "score": 0,
            "streak_best": 0,
            "sessions": 0
        },
        "last_topic": None,
        "last_difficulty": "Easy",
    }

def save_profile(name: str, prof: dict):
    prof["last_seen"] = time.time()
    with profile_path(name).open("w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)

def step_diff(diff_name: str, harder: bool, easier: bool) -> str:
    i = DIFFS.index(diff_name)
    if easier and i > 0:  i -= 1
    if harder and i < 2:  i += 1
    return DIFFS[i]

def pick_question(topic: str, difficulty: str, asked_ids: set):
    if topic == "General Science":
        pool = BANK[
        (BANK["difficulty"] == difficulty) &
        (~BANK["id"].isin(asked_ids))
    ]
    else:
        pool = BANK[
        (BANK["topic"] == topic) &
        (BANK["difficulty"] == difficulty) &
        (~BANK["id"].isin(asked_ids))
    ]
    if pool.empty:
        return None
    row = pool.sample(1).iloc[0]
    
    return {
    "id": str(row["id"]),
    "text": str(row["question"]),
    "options": list(row["options_list"]),
    "correct_idx": "ABCD".index(str(row["correct_option"]).strip()[0].upper()),
    "explanation": str(row.get("explanation", "") or ""),
    "topic": str(row.get("topic", "General Science") or "General Science"),
    "concept": str(row.get("concept", "Unknown") or "Unknown"),
    "prerequisites": str(row.get("prerequisites", "") or "")
}

def append_attempt(rec: dict):
    out = SUBMIT_DIR / "all_sessions.csv"
    header_needed = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","student","topic",
            "difficulty_before","difficulty_after",
            "question_id","correct","score_after","total_answered_session",
            "time_taken_sec","hint_used","session_id","concept","prerequisites","mastery_before","mastery_after","concept_action","response_type","rag_feedback"
        ])
        if header_needed: w.writeheader()
        w.writerow(rec)

def get_prereq_list(value):
    if value is None:
        return []
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def mastery_key(student, concept):
    return f"{student}::{concept}"


def get_concept_mastery(student, concept):
    key = mastery_key(student, concept)
    if "concept_mastery" not in st.session_state:
        st.session_state["concept_mastery"] = {}
    return st.session_state["concept_mastery"].get(key, 50)


def update_concept_mastery(student, concept, correct, hint_used=False):
    old = get_concept_mastery(student, concept)

    if correct and not hint_used:
        change = 10
    elif correct and hint_used:
        change = 5
    elif not correct and hint_used:
        change = -8
    else:
        change = -12

    new = max(0, min(100, old + change))
    st.session_state["concept_mastery"][mastery_key(student, concept)] = new

    return old, new
# -------------------- HINT ENGINE (ALWAYS 50/50) --------------------
# Only change in this version: hint ALWAYS reduces to 2 options (correct + 1 wrong).
def _trim(s: str, n: int = 140) -> str:
    s = (s or "").strip()
    return (s[:n-1]+"…") if len(s) > n else s

STOPWORDS = {"which","these","those","there","their","about","because","while","after","before","between",
             "during","would","could","should","where","when","what","from","with","into","through",
             "many","some","most","other","often","usually","generally"}

def extract_keywords(text: str, k: int = 3):
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text or "")
    words = [w for w in words if w.lower() not in STOPWORDS]
    words = sorted(set(words), key=len, reverse=True)
    return ", ".join(words[:k])

def build_hint(q):
    """
    ALWAYS prefer a 50/50 reduction (keep correct + one random wrong).
    Returns (hint_text, reduced_options, new_correct_idx).
    We keep the text very short; UI doesn't need to show it.
    """
    # short tip (unused visually, but kept if you want to show later)
    expl = _trim(str(q.get("explanation") or ""), 120)
    kw   = extract_keywords(q["text"])
    hint_text = expl if expl else ("Focus on: " + (kw if kw else "the key idea"))

    correct_idx = int(q["correct_idx"])
    opts = q["options"]

    if len(opts) > 2:
        wrong = [i for i in range(len(opts)) if i != correct_idx]
        keep_wrong = random.sample(wrong, 1)
        keep = sorted([correct_idx] + keep_wrong)
        reduced = [opts[i] for i in keep]
        new_correct_idx = keep.index(correct_idx)
        return hint_text, reduced, new_correct_idx

    # already 2 options
    return hint_text, None, None

# -------------------- STATE --------------------
EPISODE_LEN = 10  # questions per session

if "sess" not in st.session_state:
    topics = ["General Science","Biology","Chemistry","Physics","Earth Science"]
    st.session_state.sess = {
        "student": "Player",
        "topic": topics[0],
        "difficulty": "Easy",
        "total": 0,
        "correct": 0,
        "streak": 0,
        "score": 0,
        "asked_ids": set(),
        "qpack": None,
        "answered": False,
        "last_feedback": None,
        "session_id": None,
        "session_started_at": None,
        "hint_used": False,
        "hint_payload": None,  # (reduced_options, new_correct_idx)
        "recent_results": [] ,  
        "question_started_at": None,
        "episode_done": False,
        "last_rag_feedback": "",
        "session_attempts": [],
    }
S = st.session_state.sess

# -------------------- THEME / HEADER (centered) --------------------
st.markdown("""
<style>
.main-wrap {max-width: 980px; margin: 0 auto;}
.hero {text-align:center;}
.hero .underline{
  height:6px; width:68%; margin:8px auto 14px auto; border-radius:6px;
  background: linear-gradient(90deg,#3bd1ff,#7aff9e);
}
.sel-row .stTextInput, .sel-row .stSelectbox {width:100%;}
.hintbox{
  background: rgba(24,119,242,.14);
  border: 1px solid rgba(24,119,242,.35);
  padding: 8px 12px; border-radius: 10px; display:inline-block;
  font-size:.92rem; margin-top:6px;
}
.agent-pill{
  display:inline-block; padding:6px 10px; margin-top:6px; border-radius:14px;
  background: linear-gradient(90deg,#6ec6ff,#ffd36e);
  color:#111; font-weight:600;
}
.diff-pill{
  display:inline-block; padding:6px 10px; margin-left:6px; border-radius:14px;
  background: linear-gradient(90deg,#ffd36e,#ff8f70);
  color:#111; font-weight:600;
}
</style>
<div class="main-wrap">
  <div class="hero">
    <h1>🎮 Dynamic Game-Based Tutor</h1>
    <div class="underline"></div>
  </div>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Student", "Teacher"], index=0)

# ==================================================================
#                           STUDENT MODE
# ==================================================================
if mode == "Student":
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    # ---- Name + Topic row ----
    st.markdown('<div class="sel-row">', unsafe_allow_html=True)
    colA, colB = st.columns([1,1])
    with colA:
        prev_name = S["student"]
        S["student"] = st.text_input("Your Name", value=S["student"])
        if S["student"].strip() and S["student"] != prev_name:
            prof = load_profile(S["student"])
            S["difficulty"] = prof.get("last_difficulty", "Easy")
            if prof.get("last_topic"): S["topic"] = prof["last_topic"]
            st.toast(f"Loaded profile for {S['student']}", icon="✅")
            save_profile(S["student"], prof)
    with colB:
        topics = ["General Science",
                  "Biology", 
                  "Chemistry", 
                  "Physics", 
                  "Earth Science"
                  ]
        if S["topic"] not in topics:
            S["topic"] = "General Science"
        S["topic"] = st.selectbox(
            "Topic",
            topics,
            index=topics.index(S["topic"]),
            key="topic_selectbox"
            )  # st.markdown("</div>",unsafe_allow_html=True)

    # ---- Lifetime panel (from profile) ----
    prof = load_profile(S["student"])
    lifetime_sessions = int(prof["cumulative"]["sessions"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⭐ Score", S["score"])
    sess_acc = (S["correct"]/S["total"])*100 if S["total"]>0 else None
    c2.metric("Accuracy", f"{sess_acc:.0f}%" if sess_acc is not None else "—")
    c3.metric("🔥 Streak", S["streak"])
    c4.metric("🌿 Difficulty", S["difficulty"])
    c5.metric("📊 Sessions", lifetime_sessions)

    st.progress(min(S["total"], EPISODE_LEN) / EPISODE_LEN)

    if S["total"] >= EPISODE_LEN and not S["answered"]:
        S["episode_done"] = True

    if S["episode_done"]:
        acc = round((S["correct"] / max(1, S["total"])) * 100)
        st.success(f"🏁 Session complete! Score: **{S['score']}** • Session Accuracy: {acc}%")

        attempts = pd.DataFrame(S.get("session_attempts", []))

        if len(attempts) > EPISODE_LEN:
            attempts = attempts.tail(EPISODE_LEN).reset_index(drop=True)

        if not attempts.empty:
            total_qs = len(attempts)
            hints_used = int(attempts["hint_used"].sum()) if "hint_used" in attempts.columns else 0
            avg_time = round(attempts["time_taken_sec"].mean(), 1) if "time_taken_sec" in attempts.columns else 0
            topic_perf = attempts.groupby("topic")["correct"].mean().sort_values(ascending=False)

            strong_topics = topic_perf[topic_perf >= 0.7].index.tolist()
            weak_topics = topic_perf[topic_perf < 0.5].index.tolist()

            focus_area = weak_topics[0] if weak_topics else "N/A"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Questions", total_qs)
            m2.metric("Hints Used", hints_used)
            m3.metric("Avg Time / Q", f"{avg_time}s")
            m4.metric("Revise", focus_area)

            st.markdown("### 🧠 Learning Insights")

            topic_perf = attempts.groupby("topic")["correct"].mean().sort_values(ascending=False)

            strong_topics = topic_perf[topic_perf >= 0.7].index.tolist()
            weak_topics = topic_perf[topic_perf < 0.5].index.tolist()

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**✅ Strong Topics**")
                if strong_topics:
                    for t in strong_topics:
                        st.write(f"• {t}")
                else:
                    st.write("More practice needed to identify strong areas.")

            with c2:
                st.markdown("**⚠️ Topics Needing Reinforcement**")
                if weak_topics:
                    for t in weak_topics:
                        st.write(f"• {t}")
                else:
                    st.write("No major weak topic detected.")

            if "topic" in attempts.columns and "correct" in attempts.columns:
                st.markdown("### 📊 Topic Performance")
                topic_table = topic_perf.reset_index()
                topic_table.columns = ["Topic", "Accuracy"]
                topic_table["Accuracy"] = (topic_table["Accuracy"] * 100).round(1).astype(str) + "%"

                st.dataframe(topic_table, use_container_width=True, hide_index=True)
        colX, colY = st.columns([1,1])
        if colX.button("💾 Save Session & New"):
            prof = load_profile(S["student"])
            prof["cumulative"]["sessions"] += 1
            prof["cumulative"]["score"] += int(S["score"])
            prof["last_topic"] = S["topic"]
            prof["last_difficulty"] = S["difficulty"]
            save_profile(S["student"], prof)
            S.update({
                "difficulty": "Easy",
                "total": 0, "correct": 0, "streak": 0, "score": 0,
                "asked_ids": set(), "qpack": None, "answered": False,
                "last_feedback": None, "episode_done": False,
                "session_id": None, "session_started_at": None,
                "hint_used": False, "hint_payload": None, 
                "question_started_at": None,
                "last_rag_feedback": ""
            })
            st.rerun()
        if colY.button("🏠 Back to Start (keep lifetime)"):
            S.update({
                "difficulty": "Easy",
                "total": 0, "correct": 0, "streak": 0, "score": 0,
                "asked_ids": set(), "qpack": None, "answered": False,
                "last_feedback": None, "episode_done": False,
                "session_id": None, "session_started_at": None,
                "hint_used": False, "hint_payload": None, 
                "question_started_at": None,
                "last_rag_feedback": ""
            })
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if not S["session_id"]:
        S["session_id"] = f"{slug(S['student'])}_{int(time.time())}"
        S["session_started_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    if S["qpack"] is None:
        S["qpack"] = pick_question(S["topic"], S["difficulty"], S["asked_ids"])
        S["answered"] = False
        S["last_feedback"] = None
        S["hint_used"] = False
        S["hint_payload"] = None
        S["question_started_at"] = time.time()

    if S["qpack"] is None:
        st.info("No more questions in this topic/difficulty. Change topic or start a new session.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ---------- render question ----------
    q = S["qpack"]
    q_topic = q.get("topic", "General Science")
    q_concept = q.get("concept", "Unknown")
    q_prereq = q.get("prerequisites", "")
    prereq_list = get_prereq_list(q_prereq)
    
    st.markdown(f"### ❓ Q{S['total']+1}. {q['text']}")
    q_topic = q.get("topic", "General Science")
    st.caption(f"📚 Topic: {q_topic}")

    if S.get("hint_payload"):
        current_opts, current_corr = S["hint_payload"]
    else:
        current_opts, current_corr = q["options"], q["correct_idx"]

    with st.form("quiz_form", clear_on_submit=False):
        choice = st.radio("Choose an option", current_opts, index=0, key="choice_radio")
        c1, c2, c3 = st.columns([1,1,1])
        submit_clicked = c1.form_submit_button("✅ Submit", disabled=S["answered"])
        hint_clicked   = c2.form_submit_button("🧠 Hint (50/50, −2)", disabled=S["hint_used"] or S["answered"])
        skip_clicked   = c3.form_submit_button("⏭️ Skip", disabled=S["answered"])

    # --- HINT: before grading; ALWAYS 50/50; re-render immediately ---
    if hint_clicked and not S["hint_used"] and not S["answered"]:
        hint_text, reduced, new_idx = build_hint(q)
        S["score"] = max(0, S["score"] - 2)
        S["hint_used"] = True
        if reduced is not None:
            S["hint_payload"] = (reduced, new_idx)   # exactly 2 options
        # keep UI clean; just show reduced options
        st.rerun()

    # --- SUBMIT ---
    if submit_clicked and not S["answered"]:
        if S.get("hint_payload"):
            opts, corr = S["hint_payload"]
        else:
            opts, corr = q["options"], q["correct_idx"]
        is_correct = (choice == opts[int(corr)])
        S["total"] += 1
        if is_correct:
            S["correct"] += 1
            S["streak"]  += 1
            S["score"]   += 10 + 2*S["streak"]
            if S["streak"] >= 3:
                st.balloons()
        else:
            S["streak"] = 0
            S["score"]  = max(0, S["score"] - 2)
            q = S["qpack"]

        q_topic = q.get("topic", "General Science")
        q_concept = q.get("concept", "Unknown")
        q_prereq = q.get("prerequisites", "")
        prereq_list = get_prereq_list(q_prereq)

        correct_answer = opts[int(corr)]

        mastery_target = q_topic

        student_name = S.get("student", S.get("player", "Player"))
        old_mastery=50
        new_mastery=50    
        old_mastery, new_mastery = update_concept_mastery(
            student_name,
            mastery_target,
            is_correct,
            S.get("hint_used", False)
        )
        if new_mastery > old_mastery:
            mastery_status = "Topic understanding improving"
        elif new_mastery < old_mastery:
            mastery_status = "Topic needs reinforcement"
        else:
            mastery_status = "Stable understanding"

        S["last_mastery"] = {
            "topic": q_topic,
            "concept": q_concept,
            "old": old_mastery,
            "new": new_mastery,
            "status": mastery_status
}



        old_diff = S["difficulty"]
        action   = "harder" if is_correct else "easier"

        if new_mastery > old_mastery:
            mastery_status = "Concept improving"
        elif new_mastery < old_mastery:
            mastery_status = "Needs reinforcement"
        else:
            mastery_status = "Stable understanding"

        if is_correct:
            concept_action = "Advance / increase difficulty"
        else:
            if prereq_list:
                concept_action = f"Remediate prerequisite: {prereq_list[0]}"
            else:
                concept_action = "Reinforce same concept"
        if not is_correct:
            with st.spinner("🤖 AI Tutor is preparing explanation..."):
                S["last_rag_feedback"] = rag_tutor.explain_wrong_answer(
                    question=q.get("text", ""),
                    selected_answer=choice,
                    correct_answer=correct_answer,
                    topic=q_topic,
                    concept=q_concept,
                    prerequisites=q_prereq
            )
        else:
            S["last_rag_feedback"] = ""
    
        S["recent_results"].append(1 if is_correct else 0)
        S["recent_results"] = S["recent_results"][-3:]

        recent = S["recent_results"]

        if len(recent) < 3:
            new_diff = old_diff
            action = "collecting evidence"
        elif sum(recent) == 3:
            new_diff = step_diff(old_diff, harder=True, easier=False)
            action = "harder"
        elif recent.count(0) >= 2:
            new_diff = step_diff(old_diff, harder=False, easier=True)
            action = "easier"
        else:
            new_diff = old_diff
            action = "same"
            
        S["difficulty"] = new_diff
        S["asked_ids"].add(q["id"])
        S["answered"] = True
        S["last_feedback"] = ("✅ Correct!" if is_correct else "❌ Wrong!", old_diff, action, new_diff)

        prof = load_profile(S["student"])
        prof["cumulative"]["total"] += 1
        if is_correct: prof["cumulative"]["correct"] += 1
        prof["cumulative"]["streak_best"] = max(prof["cumulative"]["streak_best"], S["streak"])
        prof["last_topic"] = S["topic"]; prof["last_difficulty"] = S["difficulty"]
        save_profile(S["student"], prof)

        time_taken = max(0.0, time.time() - (S["question_started_at"] or time.time()))
        if time_taken <= 3:
            response_type = "Fast guess"
        elif time_taken <= 8:
            response_type = "Calculated attempt"
        else:
            response_type = "Thoughtful attempt"

        append_attempt({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "student": S["student"],
            "topic": S["topic"],
            "difficulty_before": old_diff,
            "difficulty_after": new_diff,
            "question_id": q["id"],
            "correct": int(is_correct),
            "score_after": S["score"],
            "total_answered_session": S["total"],
            "time_taken_sec": round(time_taken, 1),
            "hint_used": int(S["hint_used"]),
            "session_id": S["session_id"],
            "concept": q_concept,
            "prerequisites": q_prereq,
            "mastery_before": old_mastery,
            "mastery_after": new_mastery,
            "concept_action": concept_action,
            "response_type": response_type,
            "rag_feedback": S.get("last_rag_feedback", "")})
        S.setdefault("session_attempts", [])

        S["session_attempts"].append({
            "topic": q_topic,
            "concept": q_concept,
            "correct": int(is_correct),
            "time_taken_sec": round(time_taken, 1),
            "hint_used": int(S["hint_used"]),
            "response_type": response_type,
            "rag_feedback": S.get("last_rag_feedback", "")
        })       


    # --- SKIP ---
    if skip_clicked and not S["answered"]:
        S["streak"] = 0
        S["score"]  = max(0, S["score"] - 1)
        old_diff    = S["difficulty"]
        new_diff    = step_diff(old_diff, harder=False, easier=True)
        S["difficulty"] = new_diff
        S["asked_ids"].add(q["id"])
        S["answered"] = True
        S["last_feedback"] = ("⏭️ Skipped", old_diff, "easier", new_diff)

        prof = load_profile(S["student"])
        prof["cumulative"]["total"] += 1
        prof["last_topic"] = S["topic"]; prof["last_difficulty"] = S["difficulty"]
        save_profile(S["student"], prof)

        time_taken = max(0.0, time.time() - (S["question_started_at"] or time.time()))
       

        append_attempt({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "student": S["student"],
            "topic": S["topic"],
            "difficulty_before": old_diff,
            "difficulty_after": new_diff,
            "question_id": q["id"],
            "correct": 0,
            "score_after": S["score"],
            "total_answered_session": S["total"],
            "time_taken_sec": round(time_taken, 1),
            "hint_used": int(S["hint_used"]),
            "session_id": S["session_id"],
            "concept": q.get("concept", "Unknown"),
            "prerequisites": q.get("prerequisites", ""),
            "mastery_before": "",
            "mastery_after": "",
            "concept_action": "Skipped question",
            "response_type": "Skipped",
        })

    # --- feedback + nav ---
    if S["answered"] and S["last_feedback"]:
        verdict, od, act, nd = S["last_feedback"]
        (st.success if "✅" in verdict else st.error)(verdict)
        st.markdown(
            f'<span class="agent-pill">Agent action: {act}</span>'
            f'<span class="diff-pill">Difficulty: {od} → {nd}</span>',
            unsafe_allow_html=True
        )
        last_mastery = S.get("last_mastery", {})

        display_topic = last_mastery.get("topic", "General Science")
        display_concept = last_mastery.get("concept", "Unknown")

        old_mastery = last_mastery.get("old", 50)
        new_mastery = last_mastery.get("new", 50)

        mastery_status = last_mastery.get(
            "status",
            "Stable understanding"
        )

        st.markdown(
            f"""
            <span class="agent-pill">📊 Topic Mastery: {display_topic}</span>
            <span class="diff-pill">🎯 Mastery: {old_mastery}% → {new_mastery}%</span>
            """,
            unsafe_allow_html=True
        )

        st.caption(f"🧠 Current concept: {display_concept}")

        st.caption(f"💬 Mastery status: {mastery_status}")

        if new_mastery > old_mastery:
            st.caption(
        "📈 After answering this question, your estimated understanding of this concept improved."
        )
        elif new_mastery < old_mastery:
            st.caption(
        "📉 Your responses suggest this concept may need more reinforcement and practice."
        )
        else:
            st.caption(
        "📘 Your estimated understanding of this concept remained stable."
        )
        if S.get("last_rag_feedback"):
            st.markdown("### 🤖 AI Tutor Explanation")
            st.info(S["last_rag_feedback"])
        colN, colR = st.columns([1,1])
        if colN.button("⏭️ Next", disabled=not S["answered"]):

            if S["total"] >= EPISODE_LEN:
                S["episode_done"] = True
            else:
                S["qpack"] = None
                S["answered"] = False
                S["last_feedback"] = None
                S["hint_used"] = False
                S["hint_payload"] = None
                S["last_rag_feedback"] = ""
                S["question_started_at"] = None

            st.rerun()

        if colR.button("🔁 Reset Session"):
            S.update({
                "difficulty": "Easy",
                "total": 0, "correct": 0, "streak": 0, "score": 0,
                "asked_ids": set(), "qpack": None,
                "answered": False, "last_feedback": None,
                "episode_done": False, "session_id": None,
                "session_started_at": None, "hint_used": False,
                "hint_payload": None, 
                "question_started_at": None,
                "last_rag_feedback": ""
            })
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # /main-wrap

# ==================================================================
#                          TEACHER MODE
# ==================================================================
else:

    st.subheader("👩‍🏫 Teacher Dashboard")

    all_csv = SUBMIT_DIR / "all_sessions.csv"

    if all_csv.exists():
        df = pd.read_csv(all_csv)
    else:
        df = pd.DataFrame()

    if df.empty:
        st.info("No student activity available yet.")
        st.stop()

    # =========================================================
    # STRONG TOP METRICS
    # =========================================================

    total_students = df["student"].nunique()
    class_accuracy = round(df["correct"].mean() * 100, 1)

    student_acc = df.groupby("student")["correct"].mean() * 100
    at_risk_students = int((student_acc < 50).sum())

    gap_counts = df[df["correct"] == 0]["concept"].value_counts()
    top_gap_count = int(gap_counts.iloc[0]) if not gap_counts.empty else 0
    top_gap_name = gap_counts.index[0] if not gap_counts.empty else "None"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students", total_students)
    m2.metric("Class Acc.", f"{class_accuracy}%")
    m3.metric("At Risk", at_risk_students)
    m4.metric("Gap Count", top_gap_count)

    st.caption(f"⚠️ Top Knowledge Gap: {top_gap_name}")

    st.markdown("---")

    # =========================================================
    # STUDENT PERFORMANCE
    # =========================================================

    st.markdown("### 📊 Student Performance")

    perf = (
        df.groupby("student")
        .agg(
            Accuracy=("correct", "mean"),
            Questions=("correct", "count"),
            AvgScore=("score_after", "mean")
        )
        .reset_index()
    )

    perf["Accuracy"] = (perf["Accuracy"] * 100).round(1)
    perf["AvgScore"] = perf["AvgScore"].round(1)

    perf = perf.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    perf.index = perf.index + 1

    st.dataframe(perf, use_container_width=True)

    # =========================================================
    # KNOWLEDGE GAP DETECTION
    # =========================================================

    st.markdown("### 🧠 Knowledge Gap Detection")

    gap_df = (
        df[df["correct"] == 0]
        .groupby(["topic", "concept"])
        .size()
        .reset_index(name="Mistakes")
        .sort_values("Mistakes", ascending=False)
    )

    if gap_df.empty:
        st.success("No major knowledge gaps detected yet.")
    else:
        gap_view = gap_df.head(15).reset_index(drop=True)
        gap_view.index = gap_view.index + 1
        st.dataframe(gap_view, use_container_width=True)

# =========================================================
# DIFFICULTY PROGRESSION ANALYTICS
# =========================================================

    st.markdown("### 🚀 Difficulty Progression Analytics")

    difficulty_order = {
        "Easy": 1,
        "Medium": 2,
        "Hard": 3
    }

    df_progress = df.copy()

    df_progress["difficulty_score"] = df_progress["difficulty_after"].map(difficulty_order)

    progress_data = (
        df_progress.groupby("total_answered_session")
        .agg(
            AvgDifficulty=("difficulty_score", "mean"),
            Accuracy=("correct", "mean")
        )
        .reset_index()
    )

    progress_data["Accuracy"] = (progress_data["Accuracy"] * 100).round(1)

    chart = (
        alt.Chart(progress_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("total_answered_session:Q", title="Question Number"),
            y=alt.Y(
                "AvgDifficulty:Q",
                title="Average Difficulty",
                scale=alt.Scale(domain=[1, 3])
            ),
            tooltip=["total_answered_session", "AvgDifficulty", "Accuracy"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)

    difficulty_labels = {
        1: "Easy",
        2: "Medium",
        3: "Hard"
    }

    progress_view = progress_data.copy()

    progress_view["Difficulty Level"] = progress_view["AvgDifficulty"].round().map(difficulty_labels)

    progress_view = progress_view.rename(columns={
        "total_answered_session": "Question No",
        "Accuracy": "Class Accuracy %"
    })

    progress_view = progress_view[
        ["Question No", "Difficulty Level", "Class Accuracy %"]
    ]

    progress_view.index = progress_view.index + 1

    st.dataframe(progress_view, use_container_width=True)

    # =========================================================
    # ADAPTIVE DIFFICULTY FLOW
    # =========================================================

    st.markdown("### 🎯 Adaptive Difficulty Flow")

    diff_flow = (
        df.groupby(["difficulty_before", "difficulty_after"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    diff_flow = diff_flow.reset_index(drop=True)
    diff_flow.index = diff_flow.index + 1

    st.dataframe(diff_flow, use_container_width=True)

    # =========================================================
    # RAG TUTOR FEEDBACK
    # =========================================================

    if "rag_feedback" in df.columns:

        st.markdown("### 🤖 RAG Tutor Feedback")

        rag_view = df[
            ["student", "topic", "concept", "rag_feedback"]
        ].dropna()

        rag_view = rag_view[
            rag_view["rag_feedback"].astype(str).str.len() > 10
        ]

        if rag_view.empty:
            st.info("No RAG feedback generated yet. It appears after wrong answers.")
        else:
            rag_view = rag_view.tail(5).reset_index(drop=True)
            rag_view.index = rag_view.index + 1
            st.dataframe(rag_view, use_container_width=True)

    # =========================================================
    # SESSION-WISE SUMMARY
    # =========================================================

    st.markdown("### 🧾 Session Summary")

    session_summary = (
        df.groupby(["student", "session_id"], as_index=False)
        .agg(
            Questions=("correct", "count"),
            Accuracy=("correct", "mean"),
            FinalScore=("score_after", "max"),
            AvgTime=("time_taken_sec", "mean"),
            HintsUsed=("hint_used", "sum"),
            LastActivity=("timestamp", "max")
        )
    )

    session_summary["Accuracy"] = (session_summary["Accuracy"] * 100).round(1).astype(str) + "%"
    session_summary["FinalScore"] = session_summary["FinalScore"].round(1)
    session_summary["AvgTime"] = session_summary["AvgTime"].round(1).astype(str) + "s"

    session_summary = session_summary.sort_values("LastActivity", ascending=False).reset_index(drop=True)
    session_summary.index = session_summary.index + 1

    st.dataframe(session_summary, use_container_width=True)