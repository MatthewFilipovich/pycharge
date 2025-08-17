# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyCharge"
copyright = "2021-2025, Matthew Filipovich"
author = "Matthew Filipovich"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

autoapi_member_order = "alphabetical"
autoapi_dirs = ["../../pycharge"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autodoc_typehints = "description"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": ["examples", "quickstart", "user_guide"],
    "gallery_dirs": ["auto_examples", "auto_quickstart", "auto_user_guide"],
    "reference_url": {"pycharge": None},
    "filename_pattern": "^((?!sphinx_skip).)*$",  # Exclude files with 'sphinx_skip' in the name
    "matplotlib_animations": (True, "jshtml"),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "PyCharge"

# html_favicon = "_static/favicon.png"
# html_logo = "_static/favicon.png"
html_js_files = ["custom-icon.js"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MatthewFilipovich/pycharge",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pycharge",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "sg_download_links", "sg_launcher_links"],
    "logo": {"text": "PyCharge"},
}

html_static_path = ["_static"]
html_sidebars = {"auto_quickstart/quickstart": []}  # Disable sidebar for specific pages


# -- Custom configuration ----------------------------------------------------
# Skip modules in the autoapi extension to avoid duplication errors
def skip_modules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_modules)
    app.add_css_file("hide_links.css")  # Custom CSS to hide jupyter links
