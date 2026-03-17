# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Python tutorial scripts that need converting to Markdown
PY_TUTORIALS  := $(wildcard tutorials/*.py)
MD_TUTORIALS  := $(PY_TUTORIALS:.py=.md)

# Convert .py tutorials to .md
tutorials/%.md: tutorials/%.py py2md.py
	python py2md.py $< -o $@

# Convenience target to convert all tutorials
convert-tutorials: $(MD_TUTORIALS)

clean-tutorials:
	rm -f $(MD_TUTORIALS)

# Standard Sphinx targets — convert tutorials first
SPHINX_TARGETS := html dirhtml singlehtml pickle json htmlhelp qthelp \
                  devhelp epub latex latexpdf text man texinfo gettext \
                  changes linkcheck doctest coverage xml pseudoxml clean

.PHONY: help convert-tutorials clean-tutorials $(SPHINX_TARGETS)

help: convert-tutorials
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(SPHINX_TARGETS): convert-tutorials
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
