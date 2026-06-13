"""Post-Mortem page — grades historical recommendations against current prices."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.title("🪞 Post-Mortem")
    st.caption("How did past recommendations turn out? Audit trail vs. current prices.")

    days_window = st.selectbox("Trailing window", [30, 90, 180, 365], index=1)
    with_narrator = st.checkbox("Add LLM-written calibration assessment (~1 extra DeepSeek call)", value=False)
    if st.button("Generate report", type="primary"):
        from src.storage.postmortem import generate_report
        with st.spinner(f"Fetching current prices and grading {days_window}d of audit history…"):
            try:
                st.session_state["post_mortem"] = generate_report(
                    since_days=int(days_window),
                    with_narrator=with_narrator,
                )
            except Exception as e:
                st.error(f"Post-mortem failed: {e}")

    pm = st.session_state.get("post_mortem")
    if pm is None:
        st.info("Press the button above to run.")
        return

    st.markdown(
        f"**Period:** {pm.period_start.date()} → {pm.period_end.date()}  ·  "
        f"**{pm.total_recommendations}** recommendations"
    )
    st.markdown(f"**Summary:** {pm.summary}")

    if pm.by_type:
        st.markdown("### Calibration by recommendation type")
        type_rows = []
        for rec_type, stats in pm.by_type.items():
            decided = stats.correct + stats.wrong
            type_rows.append({
                "Type": rec_type,
                "Total": stats.total,
                "Correct": stats.correct,
                "Wrong": stats.wrong,
                "Pending": stats.pending,
                "Hit rate": f"{stats.hit_rate*100:.0f}%" if decided else "—",
            })
        st.dataframe(pd.DataFrame(type_rows), hide_index=True, width="stretch")

    col_w, col_l = st.columns(2)
    with col_w:
        st.markdown("### ✅ Notable winners")
        if pm.notable_winners:
            for o in pm.notable_winners:
                st.write(f"• `{o.symbol}` {o.recommendation_type} → **{o.pct_change:+.1f}%** in {o.days_elapsed}d")
                st.caption(o.explanation)
        else:
            st.write("_(none yet)_")
    with col_l:
        st.markdown("### ❌ Notable losers")
        if pm.notable_losers:
            for o in pm.notable_losers:
                st.write(f"• `{o.symbol}` {o.recommendation_type} → **{o.pct_change:+.1f}%** in {o.days_elapsed}d")
                st.caption(o.explanation)
        else:
            st.write("_(none yet)_")

    if pm.outcomes:
        with st.expander("All outcomes (full table)", expanded=False):
            rows = [{
                "Symbol": o.symbol,
                "Date": o.timestamp.date().isoformat(),
                "Type": o.recommendation_type,
                "Then": f"{o.price_at_recommendation:.2f}",
                "Now": f"{o.current_price:.2f}",
                "Days": o.days_elapsed,
                "%": f"{o.pct_change:+.1f}%",
                "Outcome": o.outcome,
                "Why": o.explanation,
            } for o in pm.outcomes]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
