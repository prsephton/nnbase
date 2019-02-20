This is a simple backpropagation neural net library for C++.

nnbase.cc/.h:  Basic functions for creating in-memory structures representing a neural network.
               Includes some general learning algorithm tweaks, such as counterweights,
               delta-bar-delta and weight decay.  In this implementation, the algorithm
               used for learning is a function of the neurode.  One can mix and match as
               needed.  nnbase defines classes for defining networks, with ability to stream
               the definition to stdout or read a definition from stdin.
nnif.cc/.h:    A functional wrapper over nnbase, which allows for loading or saving to files,
			   iterating over values, analysing, reporting and manipulating the structure
			   programatically.	Each nnif instance embeds a single neural network, and it
			   is possible to build some complex structures by chaining nnif instances.