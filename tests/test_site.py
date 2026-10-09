"""The demo college website and its AI chat button."""
import csv
import re

from backend.rag.knowledge_base import KB_DIR


def page(client, path="/"):
    r = client.get(path)
    assert r.status_code == 200
    return r.text


def test_homepage_is_the_college_website_with_the_ai_button(client):
    html = page(client)
    assert "Greenfield Institute of Technology" in html
    assert 'id="ai-btn"' in html and "College AI Assistant" in html
    assert 'id="ai-panel"' in html and 'id="ai-frame"' in html


def test_whole_website_is_inert_so_nothing_on_it_is_clickable(client):
    html = page(client)
    site = html[html.index('<div class="site"'):html.index("<!-- AI assistant")]
    assert '<div class="site" inert>' in site
    assert not re.search(r"<a\b|href=|<button|<form|<input|onclick", site, re.I)


def test_ai_button_and_panel_are_outside_the_inert_area(client):
    html = page(client)
    site = html[html.index('<div class="site"'):html.index("<!-- AI assistant")]
    assert site.count("<div") == site.count("</div>")  # .site is closed before the widget starts
    assert html.index("<!-- AI assistant") < html.index('id="ai-btn"')


def test_chat_page_still_served_for_the_panel(client):
    html = page(client, "/assistant.html")
    assert "College AI Assistant" in html and 'id="chat-form"' in html
    assert page(client, "/site.js") and page(client, "/site.css") and page(client, "/script.js")


def test_panel_loads_the_chat_page(client):
    assert 'assistant.html' in page(client, "/site.js")


def test_website_facts_match_the_knowledge_base(client):
    html = page(client)
    with (KB_DIR / "departments.csv").open(encoding="utf-8") as f:
        for dept in csv.DictReader(f):
            assert dept["head_of_department"] in html, dept["name"]
            assert dept["name"] in html
    placements = (KB_DIR / "placements.csv").read_text(encoding="utf-8")
    assert "82 percent" in placements and "82%" in html
    assert "18 LPA" in placements and "18 LPA" in html
    assert "6.2 LPA" in placements and "6.2 LPA" in html
    students = sum(1 for _ in csv.DictReader((KB_DIR / "students.csv").open(encoding="utf-8")))
    assert f"{students:,}" in html  # "1,500"


def test_website_never_shows_the_prompt_injection_notice(client):
    assert "IGNORE ALL PREVIOUS" not in page(client).upper()
