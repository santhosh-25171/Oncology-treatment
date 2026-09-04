import os
import sys

# Entry point forwarding to integration/dashboard/app.py
dashboard_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
with open(dashboard_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), dashboard_app, 'exec')
    exec(code)

