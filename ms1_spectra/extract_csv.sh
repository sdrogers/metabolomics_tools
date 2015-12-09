#!/bin/bash

java -jar alignment.jar -d mzmatch.experimental.PeakMLToCsv ../mzXML/Day0 -rtWindow 10
java -jar alignment.jar -d mzmatch.experimental.PeakMLToCsv ../mzXML/Day1 -rtWindow 10
java -jar alignment.jar -d mzmatch.experimental.PeakMLToCsv ../mzXML/Day2 -rtWindow 10
java -jar alignment.jar -d mzmatch.experimental.PeakMLToCsv ../mzXML/Day3 -rtWindow 10
