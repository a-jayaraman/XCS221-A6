#!/usr/bin/env python
import argparse, sys, io, pathlib, os

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Converts .pytex files into .tex files..')
  parser.add_argument('infile', help="The name of the source pytex file.")
  parser.add_argument('outfile', help="The name of the destination tex file.")

  infile = parser.parse_args().infile
  outfile = parser.parse_args().outfile

  # Read the infile
  with open(infile, 'r') as f:
    file_string = ''.join(f.readlines())

  # If the file contins nothing, create the output and return
  if not len(file_string):
    pathlib.Path(outfile).touch()
    exit(0)
    
  # Split the file-string on 🐍 tags
  split_string = file_string.split('🐍')
  # Ensure there are an odd number of results (meaning there were an even
  # number of 🐍 tags)
  if len(split_string) % 2 == 0:
    raise ValueError('Unmatched 🐍 tags.')
  # Separate the non-executable strings from executable strings (every other,
  # starting with the second)
  # Reverse the order if the first character is 🐍
  if file_string[0] != '🐍':
    executable_strings = split_string[1::2]
    non_executable_strings = split_string[::2]
  else:
    executable_strings = split_string[::2]
    non_executable_strings = split_string[1::2]
  # Execute each string in a shared global/local namespace.  Save the stdout for
  # each execution.
  global_scope = {}
  local_scope = {}
  executable_results = []
  for exec_string in executable_strings:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
      exec(exec_string, global_scope, local_scope)
    except:
      raise 
    finally:
      executable_results.append(redirected_output)
      sys.stdout = old_stdout
  executable_results = [res.getvalue() for res in executable_results]
  # Join the non-executable strings back with the stdout from the execuatable
  # Remember to reverse the order if the first character of the file is 🐍
  # strings, inserting the results where the corresponding executable string was
  # originally.
  if file_string[0] != '🐍':
    # non-executable string first
    result = [x for y in zip(non_executable_strings, executable_results) for x in y]
    if len(non_executable_strings) != executable_strings:
      result.append(non_executable_strings[-1])
  else:
    # executable result first
    result = [x for y in zip(executable_results, non_executable_strings) for x in y]
    if len(non_executable_strings) != executable_strings:
      result.append(executable_strings[-1])
  with open(outfile, 'w') as f:
    f.writelines(result)

"""
Highlight with the following .sublime-syntx file:
%YAML 1.2
---
file_extensions: [pytex]
scope: pytex

contexts:
  main:
    - include: Packages/LaTeX/LaTeX.sublime-syntax
    - match: 🐍
      embed: Packages/Python/Python.sublime-syntax
      escape: 🐍
"""
