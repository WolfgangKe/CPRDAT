# CPRDAT 0.1


The **C**ardio**P**ulmonary **R**esuscitation **D**ata **A**nalysis **T**oolbox is a software tool box developed to allow data analysis of defibrillator records of cardiopulmonary resuscitation attempts. The code originates from the project "A machine learning approach towards data-driven cardiopulmonary resuscitation” at the University of Graz.

So far this toolbox contains two algorithms.
Algorithm 1 automatically detects chest compression episodes in accelerometer signals from cardiopulmonary resuscitation (CPR) attempts. 
Algorithm 2 is a Machine-Learning classifier to detect the circulation state of a patient based on Accelerometry and ECG data.
Further tools will be added later.  Exemplary data to illustrate the functionalities of the toolbox is also provided in this repository. 

## Requirements

The code is written for Python 3. No dedicated installation is needed for the program, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/) 
* [pandas](https://pypi.org/project/pandas/)
* [joblib](https://joblib.readthedocs.io/en/latest/)
* [antropy](https://github.com/raphaelvallat/antropy)

## Getting started

### Chest compression detection
Five exemplary resuscitation cases with manual annotations of experienced physicians are provided in the subfolder `cc-periods_cases`. 
Just run the Jupyter-Notebook `CC-periods.ipynb` and choose a case.

### Circulation state detection
Five exemplary resuscitation cases with manual annotations of experienced physicians are provided in the subfolder `cc-periods_cases`. 
Just run the Jupyter-Notebook `AccCircClassification.ipynb` and choose a case.

## Known issues

* Since the unit of our accelerometer data is not known, the keyword argument `cpr_thresh` in` find_pause(time, acc)` probably needs to be adapted if you use your own accelerometer data.

## Authors

* **Wolfgang J. Kern** w.kern@uni-graz.at
* **Simon Orlob** simon.orlob@medunigraz.at
* **Martin Holler** martin.holler@uni-graz.at 

W. J. Kern and M. Holler are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en). S.Orlob is affillated at Medical University of Graz

## Publications

If you find this tool useful, please cite the following associated publication.

* S. Orlob, W.J. Kern, B.Alpers, M. Schörghuber, A. Bohn, M. Holler, J.-T. Gräsner, J. Wnent, Chest compression fraction calculation: A new, automated method to precisely identify periods of chest compressions from defibrillator data., Resuscitation, vol. 172, pp. 162–169, (2022). https://doi.org/10.1016/j.resuscitation.2021.12.028
* W.J. Kern, S. Orlob, B.Alpers, M. Schörghuber, A. Bohn, M. Holler, J.-T. Gräsner, J. Wnent, A sliding-window based algorithm to determine the presence of chest compressions from acceleration data.  Data Brief, vol. 41, Art. no. 107973, (2022). https://doi.org/10.1016/j.dib.2022.107973
* W.J. Kern, S. Orlob, A. Bohn, W. Toller, J.-T. Gräsner, J. Wnent, M. Holler, Accelerometry-based classification of circulatory states during out-of-hospital cardiac arrest. IEEE-Transactions on Biomedical Engineering (In Press) https://doi.org/10.1109/TBME.2023.3242717

as well as the data repository itself:
[![DOI](https://zenodo.org/badge/488608426.svg)](https://zenodo.org/badge/latestdoi/488608426)


## Acknowledgements

MH and WK acknowledge funding of the University of Graz within the 
project "A machine learning approach towards data-driven cardiopulmonary
 resuscitation”

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.



