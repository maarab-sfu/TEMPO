"""Run from project root after extracting ZIP; idempotent, creates backups."""
from pathlib import Path
root=Path(__file__).resolve().parent
main=root/'src/api/main.py';html=root/'src/frontend/index.html'
if not main.is_file() or not html.is_file():
    raise SystemExit('Expected src/api/main.py and src/frontend/index.html; no changes made')
old=main.read_text(encoding='utf-8'); page=html.read_text(encoding='utf-8')
new=old
if 'from src.api.object_query import router as object_query_router' not in old:
    new += '\n# M5.4 experimental object-aware endpoint\nfrom src.api.object_query import router as object_query_router\napp.include_router(object_query_router)\n'
new_page=page
if 'object_mode.js' not in page:
    if '</body>' not in page:raise SystemExit('No </body> in frontend; no changes made')
    new_page=page.replace('</body>','<script src="/static/object_mode.js"></script>\n</body>')
if new!=old:
    (main.with_suffix('.py.m54bak')).write_text(old,encoding='utf-8')
    main.write_text(new,encoding='utf-8')
if new_page!=page:
    (html.with_suffix('.html.m54bak')).write_text(page,encoding='utf-8')
    html.write_text(new_page,encoding='utf-8')
print('M5.4 installed. Existing /query endpoint unchanged.')
