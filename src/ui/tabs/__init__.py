"""Streamlit UI tabs — one module per tab.

Each tab exposes a single `render()` function that draws its
content under the active tab. The thin shell in streamlit_app.py
just calls them in order:

    with tab_chat:      chat.render()
    with tab_inventory: inventory.render()
    with tab_ragas:     ragas.render()
    with tab_index:     index.render()

Keeping each tab in its own module means:
  - one place per feature to look at (no 480-line monolith),
  - clear public API (`render()`) per tab,
  - streamlit_app.py stays < 50 lines and easy to skim.

These modules are Streamlit-dependent on purpose — they call
`st.*` directly because Streamlit's render model is fundamentally
imperative. The split is for *file organisation*, not for test
isolation. Pure logic (HTTP, parsing, intent pickers) is
already extracted in src/ui/api_client.py and src/rag/intents.py.
"""

from src.ui.tabs import chat, index, inventory, ragas

__all__ = ["chat", "index", "inventory", "ragas"]
