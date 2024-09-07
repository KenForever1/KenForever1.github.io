from datetime import datetime
from material.plugins.blog.structure import Archive

"""
def on_page_markdown(markdown, *, page, config, files):
    if isinstance(page, Archive):
        page.meta["template"] = "my-custom-template.html"
"""

def on_config(config, **kwargs):
    config.copyright = f"版权所有 © 2023-{datetime.now().year}"
