# ArcSDM tools by arto-dev
Spatial Data Modeler 5 for ArcGis pro<Br>

## Changed tools and other files <br>

Standard toolbox of ArcSDM 5 works on ArcGis Desktop 10.3-10.7.1, however Experimental toolbox requires components that cannot be installed on ArcGis desktop and doesn't work. ArcGis pro is supported from version 2.5+ forward.

### Python files <br>

#### Calculate Weights (calculateweights.py) 19.5.2020<br>
1. Obsolete attributes sys.exc_type and sys.exc_value replaced by sys.exc_info ()<br>
2. If the Evidence Layer raster type is RasterBand or RasterLayer, it is converted to a Raster Dataset by deleting the last part of the raster path name. If this fails, execution is aborted. If the Data Type is RasterBand, execution would crash when calling the SearchCursor (EvidenceLayer) method.<br>
3. If the pixel type of the Evidence Raster (Input Raster) is NOT an integer, the raster name is displayed and execution is aborted on error.<br>
4. The coordinate system of the Training sites Layer must be the same as that of the Evidence Layer.<br>
5. When using FileSystem as the workspace in ArcGIS Pro (that is, writing the results to a dBase database), the field name of the database table must not be the same as the alias name of the field (case insignificant). ArcGIS Pro will crash if these names are the same. That's why I added an underscore to the end of the alias name.<br>

#### Calculate Response (calculateresponse.py) 20.5.2020<br>
1. The Input Raster Data Type cannot be RasterDataSet but RasterBand or RasterLayer. If the Data Type is RasterDataSet, execution crashes to the line “outras = arcpy.sa.Lookup (Temp_Raster," WEIGHT ")”.<br>
2. The coordinate system of the input raster must be the same as that of the Training points Layer.<br>
3. When using FileSystem (dBase database) as the workspace in ArcGIS Pro, the input data of the Calculate Response tool must have the type extension .dbf in the Input Weights Table name. The tool adds that type extension if it is missing.<br>

#### Categorical Membership (categoricalmembership.py) 22.5.2020<br>
The Reclassification parameter requires a database table that defines a classification. The Categorical & Reclass tool can create such a table, but its field names do not match that original command line (VALUE, VALUE, FMx100). ArcGIS Desktop 10.6.1 writes the FROM, TO, and OUT fields to the table. ArcGIS Pro 2.5 writes the FROM_, TO, and OUT fields. These field names have been fixed in the tool.<br>

#### Fuzzy ROC (fuzzyroc.py) 14.5.2020<br>
New tool to execute Calculate Weights, Calculate Response and ROCtool together. This tool gets two or more input rasters and one or more functions and parameters and uses same parameters to each input raster.<br>

#### Fuzzy ROC 2 (fuzzyroc2.py) 19.5.2020<br>
New tool to execute Calculate Weights, Calculate Response and ROCtool together. This tool gets two or more input rasters and one function and parameter combination to each input raster.<br>

<b>Help for both FuzzyROC tools:</b><br>
<b>Input raster names</b> – there can be any number of rasters - but at least two so far. Raster is selected either from the drop-down menu (if it found in the Contents list) or by a folder icon from a GDB database or disk (raster file).<br>
<b>Fuzzy Membership Parameters</b> – select the parameters to be used in the Fuzzy Membership tool.<br>
<b>Membership type</b> is selected from the drop-down menu. Numeric values are written in boxes.<br>
<b>Min-max values</b> are the minimum and maximum values between which Midpoint or Spread varies. The FuzzyROC tool starts with minimum values and performs calculation a count of Count times, increasing the minimum value (max - min) / count with each round.<br>
<b>Fuzzy Overlay Parameters</b> - select Overlay type from the drop-down menu. Only the Gaussian type has a parameter<br>
<b>ROC True Positives</b> Feature Class defines a feature class from which known positive cases are read.<br>
<b>ROC Destination Folder</b> - the folder where the ROC tool Output Files is written.<br>
The File Geodatabase database or file folder in the workspace is made up of a set of rasters named FM_n_m from the Fuzzy Membership tool and a set of rasters named FO_n from the Fuzzy Overlay tool.<br>
<b>FuzzyMembership.csv</b> (In the ROC Destination Folder) is an Excel spreadsheet that contains input and output data related to Fuzzy Membership tool performance.<br>
<b>FuzzyOverlay.csv</b> (In the ROC Destination Folder) is an Excel spreadsheet that contains output and output data related to Fuzzy Overlay --- tool execution.<br>
<b>FuzzyROC.csv</b> (In the ROC Destination Folder) is an Excel spreadsheet that contains the results calculated with the ROC tool.<br>

#### ROC Tool (roctool.py)	12.5.2020<br>
Added closing of pylab windows because without closing only 20 patterns could be formed.<br>

#### TOC Fuzzification (tocfuzzification.py)	29.4.2020<br>
Tested, not modified.<br>

#### Logistic Regression (logisticregression.py)	28.5.2020<br>
Logistic Regression don’t work on ArcGIS Pro if workspace is File System. Join Field cannot join raster from file system and/or DBF table fields together on ArcGIS Pro but can join them on ArcGIS Desktop 10.6.1. LR works on ArcGIS Pro if workspace is File Geodatabase and Input Weights rasters are in File Geodatabase. <br>

#### Grand Wofe (grand_wofe_lr.py)	28.5.2020<br>
1. Obsolete attributes sys.exc_type and sys.exc_value replaced by sys.exc_info ()<br>
2. Because Logistic Regression don’t work on ArcGIS Pro if workspace is File System, this tool will not work on ArcGIS Pro if workspace is File System.<br>
3. The coordinate system of the input raster must be the same as that of the Training points Layer.<br>

#### common.py 29.4.2020<br>
Addition of a result raster displayed  to ArcGIS Pro to the Contents panel in addToDisplay feature fixed.<br>

#### sdmvalues.py 28.5.2020<br>
1. The default value for Environment.Cell Size is Maximum of Inputs, which can be used as a Cell Size if the mask is FeatureLayer or FeatureClass. If the mask is a raster, the text value cannot be used in the calculation, but the Cell Size must be an integer<br>
2. RasterLayer, RasterBand, FeatureLayer and FeatureClass can be used as the mask. These types are grouped in the code into if-elif-else blocks. In the Else block, all data types other than those defined above are discarded.<br>
3. If the mask is not defined at all, you will be prompted to check the Environment settings.<br>
4. If the “Cells in area” value is 0, execution is aborted.<br>

#### workarounds_93.py 5.5.2020<br>
Obsolete attributes sys.exc_type and sys.exc_value replaced by sys.exc_info ()<br>

### Other files<br>

ArcSDM.pyt - ArcSDM Toolbox menu (Added new Fuzzy ROC tools)
ArcSDM.CalculateResponse.pyt.xml	- HELP for Calculate Response<br>
ArcSDM.CalculateWeightsTool.pyt.xml	 -HELP for Calculate Weights<br>
ArcSDM.FuzzyROC2.pyt.xml - HELP for Fuzzy ROC 2<br>
ArcSDM.LogisticRegressionTool.pyt.xml - Help for Logistic Regression<br>
ArcSDM.GrandWofe.pyt.xml - Help for Grand Wofe<br>
