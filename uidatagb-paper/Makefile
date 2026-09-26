.PHONY: all clean

# The cp step works around a local MiKTeX kpsewhich quirk that otherwise
# silently produces an empty bibliography -- see latexmkrc for details.
all:
	mkdir -p build
	cp references.bib build/references.bib
	latexmk -pdf -outdir=build main.tex

clean:
	latexmk -C -outdir=build main.tex
