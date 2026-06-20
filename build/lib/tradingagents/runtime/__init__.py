"""Launcher-era runtime: the unified `--schedule` daemon's moving parts.

These tie the per-name pipeline + macro snapshot + position overlay + ranking into
one product surface: a compact bilingual digest, a holiday-aware scheduler, and a
two-way Telegram bot. Kept out of the pure `portfolio`/agent layers so the decision
logic stays testable without network or LLM.
"""
