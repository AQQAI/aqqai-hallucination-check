"""
Live demo UI for checker.py
  pip install streamlit
  streamlit run app.py
"""
import html, json, os, time
import streamlit as st
import checker as C

st.set_page_config(page_title="Hallucination Checker", page_icon="🔍", layout="wide")

COLORS = {"HALLUCINATION": "rgba(239,68,68,0.35)", "NEEDS REVIEW": "rgba(245,158,11,0.35)",
          "OK": "rgba(34,197,94,0.25)"}
ICONS = {"HALLUCINATION": "🔴", "NEEDS REVIEW": "🟠", "OK": "🟢"}


def md(t):
    t = str(t).replace("\\", "\\\\")
    for ch in "$*_`~#[]<>|":
        t = t.replace(ch, "\\" + ch)
    return t.replace("\n", " ")

if "source" not in st.session_state:
    st.session_state.source = C.SOURCE
    st.session_state.summary = C.TESTS["added_fact"][0]

with st.sidebar:
    st.header("Demo cases")
    for name, (summ, _) in C.TESTS.items():
        if st.button(name.replace("_", " ").capitalize(), key=f"case_{name}"):
            st.session_state.source = C.SOURCE
            st.session_state.summary = summ
            st.session_state.pop("result", None)
    st.divider()
    use_hhem = st.toggle("HHEM cross-check + offline fallback", value=False)
    if os.path.exists("aggrefact_results.json"):
        s = json.load(open("aggrefact_results.json"))["summary"]
        st.divider()
        st.subheader("Measured on real data")
        st.metric("LLM-AggreFact balanced accuracy", f"{s['balanced_accuracy']:.0%}")
        st.caption(f"{s['n_scored']} human-labelled examples, seed {s['seed']}. "
                   f"Caught {s['confusion']['tn']}/{s['confusion']['tn'] + s['confusion']['fp']} unsupported, "
                   f"passed {s['confusion']['tp']}/{s['confusion']['tp'] + s['confusion']['fn']} supported. "
                   f"Accuracy 95% CI {s['accuracy_95ci'][0]:.0%}-{s['accuracy_95ci'][1]:.0%}.")

st.title("🔍 Hallucination Checker")
st.caption(f"Flags every claim in an AI summary that its source does not support · {C.MODEL}")

left, right = st.columns(2)
source = left.text_area("Source document (ground truth)", key="source", height=220)
summary = right.text_area("AI-generated summary to check", key="summary", height=220)

if st.button("Run check", type="primary"):
    status = st.status("Auditing summary against source...", expanded=True)
    status.write("1. Claude splits the summary into atomic claims and verifies each against the source")
    t0 = time.time()
    try:
        res = C.check(source, summary, use_hhem=use_hhem)
    except Exception as e:
        status.update(label="Check failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        st.session_state.pop("result", None)
        st.stop()
    claims = res["claims"]
    status.write(f"2. {len(claims)} claims verified in {time.time() - t0:.1f}s (mode: {res['mode']})")
    quoted = [c for c in claims if c["verdict"] != "NOT_IN_SOURCE" and res["mode"] == "llm"]
    bad = [c for c in quoted if "evidence quote not found in source" in c["flags"]]
    status.write(f"3. Evidence quotes found in the source: {len(quoted) - len(bad)}/{len(quoted)}")
    n_sent = len(C._sentences(summary))
    status.write(f"4. Summary sentences covered by a claim: {n_sent - len(res['unchecked_sentences'])}/{n_sent}")
    n_h = res["hallucination_count"]
    status.update(label=f"{n_h} unsupported claim(s) found" if n_h else "No unsupported claims found",
                  state="error" if n_h else "complete", expanded=False)
    st.session_state.result = {"res": res, "summary": summary, "source": source}

if "result" in st.session_state:
    res = st.session_state.result["res"]
    summary_checked = st.session_state.result["summary"]
    claims = res["claims"]
    n_h = res["hallucination_count"]
    if summary_checked != summary or st.session_state.result["source"] != source:
        st.info("Showing results for the previous text. Press Run check to check the edited text.")
    n_r = sum(c["status"] == "NEEDS REVIEW" for c in claims)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unsupported", n_h)
    m2.metric("Needs review", n_r)
    m3.metric("Supported", len(claims) - n_h - n_r)
    m4.metric("Unchecked sentences", len(res["unchecked_sentences"]))

    marked = html.escape(summary_checked)
    for c in claims:
        if c["status"] == "OK":
            continue
        span = html.escape(c["summary_span"])
        if span and span in marked:
            marked = marked.replace(
                span, f'<mark style="background:{COLORS[c["status"]]};padding:0 2px;border-radius:3px">{span}</mark>', 1)
    st.subheader("Summary with flagged claims")
    marked = marked.replace("$", "&#36;").replace("\r", "").replace("\n", "<br>")
    st.markdown(f'<div style="font-size:1.1rem;line-height:1.8">{marked}</div>', unsafe_allow_html=True)

    st.subheader("Claim-by-claim verdicts")
    order = {"HALLUCINATION": 0, "NEEDS REVIEW": 1, "OK": 2}
    for c in sorted(claims, key=lambda c: order[c["status"]]):
        with st.container(border=True):
            st.markdown(f"{ICONS[c['status']]} **{c['status']}** · `{c['verdict']}` · {md(c['claim'])}")
            st.caption(md(c["reasoning"]))
            if c.get("source_quote"):
                st.markdown(f"> Source: *{md(c['source_quote'])}*")
            for f in c["flags"]:
                st.warning(md(f))
    for sent in res["unchecked_sentences"]:
        st.warning(f"Not covered by any claim, check manually: {md(sent)}")
    with st.expander("Raw JSON"):
        st.json(res)
