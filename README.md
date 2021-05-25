This folder includes Source code (C++) and Dataset ("data" folder) of Boosting Influence Spread and Video Summarization.

To build the C++ code, run:
```
  g++ -std=c++11 *.cpp -o mp -DIL_STD -I<source code path> -fopenmp -g
```

After building, to run our code, run:
```
  ./mp 	-t 	<application type, 0: social, 1: video> 					# default: 0
		    -a 	<algorithm type, 0: FastProb, 1: ThrGreedy, 2: Greedy, 3: ResGreedy, 4: SplitGrow> 	# default: 0
		    -b 	<total budget> 									# default: 20
		    -k 	<number of groups> 								# default: 2
		    -p 	<number of threads> 								# default: 10
```
Or:
```
  ./mp 	-h
```
to print help.

The code will return the following result:
	+ Objective value
	+ Number of queries
