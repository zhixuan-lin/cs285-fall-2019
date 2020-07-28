default: makepdf clean_aux
	
makepdf: main.tex 
	latexmk -pdf -outdir='./out' main.tex
	# latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" main.tex

clean_aux:
	rm -rf ./aux *.{aux,fls,log,fdb_latexmk} _minted*
