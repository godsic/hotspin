all: manual.pdf


.PHONY: manual.pdf
manual.pdf: hotspin.tex flags.tex examples.tex *.tex
	xelatex --output-driver="xdvipdfmx -vvv" -8bit --halt-on-error manual.tex
	makeindex manual
	xelatex --output-driver="xdvipdfmx -vvv" -8bit --halt-on-error manual.tex

.PHONY: hotspin.tex
hotspin.tex:
	ln -sf ../../src/python/hotspin.tex .
	./texgen.sh

.PHONY: flags.tex
flags.tex:
	../../bin/hotspin -h 2> flags.tex

.PHONY: examples.tex
examples.tex:
	./examples.sh

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ilg *.log *.toc *.out manual.pdf *.idx *.ind modules/*.dot modules/*.pdf modules/*.aux
