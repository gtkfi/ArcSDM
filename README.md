# ArcSDM

Spatial Data Modeler 6 for ArcGIS Pro 3.0+<Br>


## How to get started? <br>

Standard toolbox of ArcSDM 5 works on ArcGis Desktop 10.3-10.7.1, however the Experimental toolbox requires components that cannot be installed on ArcGis desktop and doesn't work. ArcGIS Pro is supported from version 2.5+ forward.


If you want to work on your own data, you can download just the toolbox. If you want to try, evaluate and experiment with ArcSDM you can download our demodata separately from the main package. <br>

ArcSDM wiki contains upto date howtopage: https://github.com/gtkfi/ArcSDM/wiki/Howto-start


### Toolbox <br>
Click "Clone or download" and select the suitable download option for you. "Download Zip" is the safe and easy choice, just open the zip anywhere you like and add the toolbox to ArcGis (Desktop or Pro). <br>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=w-EAv2A2jOM
" target="_blank"><img src="http://img.youtube.com/vi/w-EAv2A2jOM/0.jpg" 
alt="How to download and extract the toolbox" width="240" height="180" border="10" /></a>


### Demodata <br>
Download the demodata as a zip package from our demodata git repository https://github.com/gtkfi/demodata. You can download the release file from here https://github.com/gtkfi/demodata/releases/download/v1.0/ArcSDM_Demodata.zip <br>
Open and save the zip optionally to your ArcSDM toolbox installation folder as "Data" folder. Then click "initworkdir.bat" to create (or overwrite older) working copy. <br>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=4rU1oDqEUrQ
" target="_blank"><img src="http://img.youtube.com/vi/4rU1oDqEUrQ/0.jpg" 
alt="How to download and extract the toolbox" width="240" height="180" border="10" /></a>


### Status
<br>
Status of the toolbox should be updated to wiki https://github.com/gtkfi/ArcSDM/wiki/Toolbox-details <bR>

## News: 

Development has started by GTK to migrate this tool for ArcGIS Pro 3.0+ (ArcSDM 6.0)

## Usage of the ArcSDM Python env

At the moment is this advised to clone the ArcGIS Pro Python env and downloading the necessary packages to the cloned Python env.
Do not clone it to the default AppData folder. This might cause the required packages to not install.
Please clone the Python env to for example C:\user\path_to_a_folder\arcpy-py3-env

[How to use Python environments in ArcGIS Pro](https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/activate-an-environment.htm)

## Setting Up ArcSDM Python Environment

To use ArcSDM with ArcGIS Pro, follow these steps:

1. **Clone the Repository**:
    - Open ArcGIS Pro.
    - Navigate to `Project` -> `Package Manager`.
    - Clone the environment (not in the default location) to your local machine.

2. **Change ArcGIS Pro to Use ArcSDM Python Environment**:
    - Set ArcGIS Pro to use the `arcsdm-py3-env` environment.
    ![Using ArcSDM Python Environment](./img/use_arcsdm-py3-env.PNG)

3. **Restart ArcGIS Pro**:
    - Close and reopen ArcGIS Pro to apply the changes.
    - Add Packages Scikit-learn (>=1.4), Tensorflow and Imbalanced-learn by searching those libraries:
    ![Add Packages to ArcSDM Python Environment](./img/add_packages_arcsdm_py3_env.PNG)
    - Sometimes packages do not install due to Proxy settings. Please retry or contact your organizations IT Help to enable the installation.

4. **Install Required Packages**:
    Go

By following these steps, you will have the necessary environment set up to use ArcSDM with ArcGIS Pro.


## History:
16.10.2020 5.03-1 merging fuzzy membership files into fuzzy overlay files rewritten<br> 
7.10.2020 5.03 fixes to calculateweights, calculateresponse, logisticregression and grandwofe<br>
14.8.2020 5.02.1 logisticregression works now on Pro 2.6 with file system workspace<br> 
13.8.2020 5.02 arto-dev branch merged to master branch.<br> 
23.7.2020 ArcSDM version 5.01.08 in the arto-dev branch.<br>
2.6.2020 arto-dev branch added. There are updated tool versions for testing.<br>
3.4.2020 New link to demodata and documentation how to run on Arcgis pro 2.5+<br>
26.4.2018 5.01.01 Merged code by Tachyon-work to master-branch.<br>
6.10.2017 5.00.22 GrandWofe and various fixes<br>
2.10.2017 Updating wiki and this page, cleaning up. <br>
4.9.2017 5.00.22 Updates, fixes and new demodata<br>
17.5.2017 5.00.15 Updates and fixes <br>
5.5.2017 5.00.14 Calculate weights error with nodata fixed <br>
4.5.2017 5.00.13 Multiple fixes for minor UI errors <br>
10.4.2017 5.00.11 Quickfix<br>
24.2.2017 5.00.10 First release of experimental SOM toolbox.<br>
28.3.2017 Update to demodata path<br>
17.2.2017 5.00.07 Fixes <br>
2.2.2017 5.00.03 First draft version of Rescale raster -tool added<br>
29.12.2016 Area Frequency Table -tool added to toolbox and tmp toolbox removed<br>
14.12.2016 Separation of demodata and toolbox started.<br>
1.11.2016 Roc-tool included <br>
1.11.2016 Calculate response and Logistic regression feature complete and ready for testing, not clean<br>
27.9.2016 Calculate response update <br>
20.9.2016 Testing started against Arcgis desktop 10.4.1 <br>
7.9.2016 Calculate response (WIP) + New demo data <br>
2.9.2016 Logistic regression (wip) Area frequency tool (wip)<br>
26.8.2016 Some new tools - Calculate weights done, needs work.<br>
8.8.2016  ArcGis desktop mxd, python toolbox - development branches <br>
26.5.2016 AddBearings, calculatebends and logistic regression tools. Demodatafixes updates<br>
28.4.2016 Logistic Regression tool (needs lots of work)<br>
18.4.2016 WofE manual steps compile<br>
13.4.2016 Demodata for tests added (from original files)<br>
1.4.2016 Repository created <br>
