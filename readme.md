# CPRDAT 0.1

The **C**ardio**P**ulmonary **R**esuscitation **D**ata **A**nalysis **T**oolbox is a software tool box developed to allow data analysis of defibrillator records of cardiopulmonary resuscitation attempts. The code originates from the project "A machine learning approach towards data-driven cardiopulmonary resuscitation” at the University of Graz.

So far this toolbox contains only one algorithm which automatically detects chest compression episodes in accelerometer signals from cardiopulmonary resuscitation (CPR) attempts. Further tools will be added later.  Exemplary data to illustrate the functionalities of the toolbox is also provided in this repository. 

## Requirements

The code is written for Python 3. No dedicated installation is needed for the program, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/) 
* [pandas](https://pypi.org/project/pandas/)

## Getting started

Five exemplary resuscitation cases with manual annotations of experienced physicians are provided in the subfolder `cc-periods_cases`. 
Just run the Jupyter-Notebook `CC-periods.ipynb` and choose a case.

## Known issues

* Since the unit of our accelerometer data is not known, the keyword argument `cpr_thresh` in` find_pause(time, acc)` probably needs to be adapted if you use your own accelerometer data.

## Authors

* **Wolfgang J. Kern** w.kern@uni-graz.at
* **Simon Orlob** simon.orlob@medunigraz.at
* **Martin Holler** martin.holler@uni-graz.at 

W. J. Kern and M. Holler are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en). S.Orlob is affillated at Medical University of Graz

## Publications

If you find this tool useful, please cite the following associated publication.

* Simon Orlob, Wolfgang J. Kern, Birgitt Alpers, Michael Schörghuber, Andreas Bohn, Martin Holler, Jan-Thorsten Gräsner, Jan Wnent,
Chest compression fraction calculation: A new, automated, robust method to identify periods of chest compressions from defibrillator data – Tested in Zoll X Series, Resuscitation, Volume 172, 2022, Pages 162-169, ISSN 0300-9572, https://doi.org/10.1016/j.resuscitation.2021.12.028. (https://www.sciencedirect.com/science/article/pii/S0300957221005360)
* Wolfgang J. Kern, Simon Orlob, Birgitt Alpers, Michael Schörghuber, Andreas Bohn, Martin Holler, Jan-Thorsten Gräsner, Jan Wnent, A sliding-window based algorithm to determine the presence of chest compressions from acceleration data, Data in Brief, Volume 41, 2022, 107973, ISSN 2352-3409, https://doi.org/10.1016/j.dib.2022.107973. (https://www.sciencedirect.com/science/article/pii/S2352340922001846)

## Acknowledgements

MH and WK acknowledge funding of the University of Graz within the 
project "A machine learning approach towards data-driven cardiopulmonary
 resuscitation”

## License
