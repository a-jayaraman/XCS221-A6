# XCS221 Student Code Repository
This repository contains all code for the corresponding assignment in XCS221.
The build tools in this repo can be used to run the autograder locally or
compile a LaTeX submission.

## Running the autograder locally
All assignment code is in the `src/` subirectory.  You will submit only the
`src/submission.py` file.  Please only make changes between the lines containing
`### START CODE HERE ###` and `### END CODE HERE ###`. Do not make changes to
files other than `src/submission.py`.

The unit tests in `src/grader.py` will be used to autograde your submission.
Run the autograder locally using the following terminal command within the
`src/` subdirectory:
```
(XCS221) $ python grader.py
```

There are two types of unit tests used by our autograders:
- `basic`:  These unit tests will verify only that your code runs without
  errors on obvious test cases.
- `hidden`: These unit tests will verify that your code produces correct
  results on complex inputs and tricky corner cases.  In the student version of
  `src/grader.py`, only the setup and inputs to these unit tests are provided.
  When you run the autograder locally, these test cases will run, but the
  results will not be verified by the autograder.

For debugging purposes, a single unit test can be run locally.  For example, you
can run the test case `3a-0-basic` using the following terminal command within
the `src/` subdirectory:
```
(XCS221) $ python grader.py 3a-0-basic
```

## Compiling a LaTeX submission
Only run `make` form the root directory.  Complete `make` documentation is
provided within the Makefile.  To get started, clone the repository and try out
a simple `make` command:
```
$ make clean -s
```
If this doesn't work, you probably don't have the LaTeX compilation management
tool that we use: `latexmk`.  Most linux distributions come pre-loaded with this
tool.  Mac users can download and install it from
[mactex](https://tug.org/mactex/).

If the command runs correctly, it will remove the assignment PDF from your root
directory.  Don't worry though!  Try recreating it again using the following
command:
```
$ make without_solutions -s
```
After some file shuffling and a few passes of the LaTeX compiler, you should see
a fresh new assignment handout in the root directory.  Now try the following
command:
```
$ make with_solutions -s
```
You should now see a *\*_Solutions.pdf* file in your root directory.  This contains
the content from the original handout as well as your solutions!  If you haven't
written any solutions yet, it will porbably look a lot like the
`without_solutions` version.

To see what it looks like with some solution code, open up any file with a name
like *\*-sol.tex.*.  Put the following code between the tags
`### START CODE HERE ###` and `### END CODE HERE ###`:
```latex
\begin{answer}
  % ### START CODE HERE ###
  \LaTeX
  % ### END CODE HERE ###
\end{answer}
```
Now run the following command:
```
$ make -s
```
This command re-runs the default `make` target, which is, conveniently,
`make with_solutions`.  Opening the file, you should see something like the
following:

<img src="https://render.githubusercontent.com/render/math?math=\LaTeX">

Good luck with the assignment!  Remember that you can always submit
organized and legible handwritten PDFs instead of typeset documents.
