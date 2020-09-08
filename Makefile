####################
# STUDENT MAKEFILE #
####################
# Possible targets:
# `make` | `make with_solutions`
#   Creates the A0_Solutions.pdf handout.  Removes auxiliary studenlatex files
#   after compilation.
# `make without_solutions`
#   Creates the A*.pdf handout.  Removes auxiliary studenlatex files after
#   compilation.  This creates the handout by creating a temporary copy of all
#   files, removing solution code, then deleting the temporary directory.
# `make clean`
#   Removes assignment handouts, both with and without solutions.

# If you want ot edit your latex live (i.e. WYSIWYG), try running the following
# command in your tex/ directory:
#
# $ latexmk -pvc -jobname="%A_Solutions" A0.tex
#
# (Of course, substituting the "A0" for the assignment root doccument)
#

SHELL = /bin/sh

THIS_ASSIGNMENT = A6

TEX_DEPENDENCIES = $(shell find tex -type f)

.DEFAULT_GOAL := with_solutions

.PHONY: clean with_solutions without_solutions

with_solutions: $(THIS_ASSIGNMENT)_Solutions.pdf

without_solutions: $(THIS_ASSIGNMENT).pdf

$(THIS_ASSIGNMENT).pdf: $(TEX_DEPENDENCIES)
	# 1. Make and move into a temp filder where the tex files can be modified (solutions removed)
	# 2. Move into the temp directory and remove solution code.
	# 3. Make the latex document and clean it up (if it compiles correctly)
	# 4. Withdraw and remove the temp directory
	mkdir -p ./temp_tex_no_solutions
	cp -a ./tex/* ./temp_tex_no_solutions/
	cd ./temp_tex_no_solutions ; \
	find "./" -name "*.tex" -exec sed -i '' '/.*START CODE HERE.*/,/.*END CODE HERE.*/{//!d;}' {} + ; \
	find "./" -name "*.tex" -exec sed -i '' '/SOLUTION ALERT/{N;d;}' {} + ; \
	latexmk -quiet $(THIS_ASSIGNMENT).tex && latexmk -quiet $(THIS_ASSIGNMENT).tex -c ; \
	cd .. ; \
	rm -rf ./temp_tex_no_solutions


$(THIS_ASSIGNMENT)_Solutions.pdf: $(TEX_DEPENDENCIES)
	# 1. Make the latex document (see tex/latexmakerc for options)
	# 2. Then clean it up (if it compiles without errors)
	cd ./tex; latexmk -quiet -jobname="%A_Solutions" $(THIS_ASSIGNMENT).tex && latexmk -quiet -jobname="%A_Solutions" $(THIS_ASSIGNMENT).tex -c

clean:
	cd ./tex; latexmk -quiet -C $(THIS_ASSIGNMENT).tex 
	cd ./tex; latexmk -quiet -jobname="%A_Solutions" -C $(THIS_ASSIGNMENT).tex 
	rm -rf ./temp_tex_no_solutions