""" ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

Area frequency tool 

Spatial Data Modeller for ESRI* ArcGIS 9.3
Copyright 2009
Gary L Raines, Reno, NV, USA: production and certification
Don L Sawatzky, Spokane, WA, USA: Python software development

Input raster is a floating raster that has no VAT.
However, can extract raster values to points.

Can get the frequency>0 of raster values from extracted points.
Because the floating raster values in the frequency table are
only those with a frequency of 1 or more,CANNOT get the remainder
of float raster values because they have no frequency.

Modified to accept an integer raster. Cannot substitute another attribute as Value
in this operation, because cannot ExtractValueToPoints for an attribute, only Value.
In case of NN output raster, user must generate a new floating raster for RBFLN, PNN,
FZZYCLSTR attributes, so proper extraction can be done.

Creates six lists, each of length equal to the VAT count.  This algorithm is designed to
operate on, say, less than 1000 training sites and 10,000 raster values.  Grids with real data
are desirable.  Images won't do.

Creates another dictionary of length equal to the Statistics table generated from raster and
training sites.

Makes five passes through the output table.  So, table cannot be long.

The floating raster VAT is a dictionary defined in floatingrasterclass.py;
with VAT[Value] = COUNT, like a VAT of an integer raster.
VAT.next() returns (ID, VALUE, COUNT)

"""

# Import modules
import sys, string, os, random, traceback, tempfile
import arcpy
import arcsdm.workarounds_93
from arcsdm.floatingrasterclass import FloatRasterVAT, rowgen
import importlib

# Create the Geoprocessor object
arcpy.CheckOutExtension("spatial")
arcpy.env.overwriteOutput = True

# Custom exception for user errors
class UserException(Exception):
    pass

# Main function to execute the tool
def Execute(self, parameters, messages):
    import arcsdm.sdmvalues
    try:
        importlib.reload(arcsdm.sdmvalues)
    except:
        importlib.reload(arcsdm.sdmvalues)
    
    # Get input parameters
    Input_point_features = parameters[0].valueAsText
    Input_raster = parameters[1].valueAsText
    Value_field = parameters[2].valueAsText
    UnitArea = parameters[3].value
    Output_Table = parameters[4].valueAsText
    
    # Append SDM values to the input point features
    arcsdm.sdmvalues.appendSDMValues(arcpy, UnitArea, Input_point_features)
    arcpy.AddMessage("\n" + "="*10 + " Starting area frequency " + "="*10)
    
    # Initialize local variables
    valuetypes = {1: 'Integer', 2: 'Float'}
    joinRastername = None
    Input_table = None
    RasterValue_field = Value_field.title().endswith('Value')
    
    # Determine the value type of the input raster
    valuetype = arcpy.GetRasterProperties_management(Input_raster, 'VALUETYPE').getOutput(0)
    arcpy.AddMessage("Valuetype = " + str(valuetype))
    
    # Handle integer raster case
    if int(valuetype) <= 8:
        if not Input_table:
            if not RasterValue_field:
                float_type = ('Double', 'Single')
                fld = arcpy.ListFields(Input_raster, Value_field)[0]
                Value_field_type = fld.type
                if Value_field_type.title() in float_type:
                    Value_field = Value_field.split('.')
                    if len(Value_field) > 1:
                        arcpy.AddError("Integer Raster has joined table.")
                        raise UserException
                    InExpression = "FLOAT(%s.%s)" % (Input_raster, Value_field[0])
                    TmpRaster = arcpy.CreateScratchName("tmp_AFT_ras", "", "raster", arcpy.env.scratchWorkspace)
                    arcpy.sa.SingleOutputMapAlgebra(InExpression, TmpRaster)
                    arcpy.AddMessage("Floating Raster from Raster Attribute: type %s" % arcpy.Describe(Input_raster).pixelType)
                else:
                    arcpy.AddError("Integer Raster Attribute field not floating type.")
                    raise UserException
            else:
                arcpy.AddError("Integer Raster Value field not acceptable.")
                raise UserException("Integer Raster Value field not acceptable")
                
        Input_raster = TmpRaster
        valuetype = 2
    else:
        arcpy.AddMessage("Floating Raster from Floating Raster Value: type %s" % arcpy.Describe(Input_raster).pixelType)
        
    # Check if there are any training points selected
    if int(arcpy.GetCount_management(Input_point_features).getOutput(0)) == 0:
        arcpy.AddError("Training Points must be selected: %s" % Input_point_features)
        raise UserException
    
    # Extract raster values to points
    Output_point_features = arcsdm.workarounds_93.ExtractValuesToPoints(Input_raster, Input_point_features, "TPFID")
    
    # Create a summary statistics table
    Output_summary_stats = arcpy.CreateScratchName("Ext_Trn_Stats", "", "Table", arcpy.env.scratchWorkspace)
    
    # Initialize a dictionary to store statistics
    stats_dict = {}
    flt_ras = FloatRasterVAT(Input_raster)
    
    # Get the raster values and their counts
    rows = flt_ras.FloatRasterSearchcursor()
    arcpy.Statistics_analysis(Output_point_features, Output_summary_stats, "RASTERVALU FIRST", "RASTERVALU")
    
    for row in rows:
        stats_dict[row.value] = 0
    num_training_sites = int(arcpy.GetCount_management(Output_point_features).getOutput(0))
    
    # Get the statistics from the summary table
    statsrows = rowgen(arcpy.SearchCursor(Output_summary_stats))
    
    num_nodata = 0
    for row in statsrows:
        if row.RASTERVALU == flt_ras.getNODATA():
            num_nodata = row.FREQUENCY
        else:
            rasval = flt_ras[row.RASTERVALU]
            if rasval in stats_dict:
                stats_dict[rasval] = row.FREQUENCY
    
    # Check if the counts match the number of training sites
    num_counts = sum(stats_dict.values())
    if num_counts != num_training_sites - num_nodata:
        arcpy.AddWarning("Stats count and number of training sites in data area do not compare.")
    if num_nodata > 0:
        arcpy.AddWarning("%d training points in NoData area." % num_nodata)
    
    # Create the output table
    arcpy.AddMessage('Creating table: %s' % Output_Table)
    fullname = arcpy.ParseTableName(Output_Table)
    database, owner, table = fullname.split(", ")
    arcpy.AddMessage('Output workspace: %s' % os.path.dirname(Output_Table))
    arcpy.AddMessage('Output table name: %s' % os.path.basename(Output_Table))
    arcpy.CreateTable_management(os.path.dirname(Output_Table), os.path.basename(Output_Table))
    arcpy.MakeTableView_management(Output_Table, 'output_table')
    
    # Add fields to the output table
    arcpy.AddField_management('output_table', "Frequency", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "RASTERVALU", "DOUBLE", "18", "8", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "Area_sqkm", "DOUBLE", "", "", "", "Area_Sq_Kilometers", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "CAPP_CumAr", "DOUBLE", "", "", "", "CAPP_Cumulative_Area", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "Eff_CumAre", "DOUBLE", "", "", "", "Efficiency_Cumulative_Area", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "Cum_Sites", "DOUBLE", "", "", "", "Cumulative_Sites", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "I_CumSites", "DOUBLE", "", "", "", "Cumulative_Sites", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management('output_table', "Eff_AUC", "DOUBLE", "", "", "", "A_U_C", "NULLABLE", "NON_REQUIRED", "")
    
    # Delete the default field
    arcpy.DeleteField_management(Output_Table, "Field1")
    
    # Calculate the area factor
    factor = (float(arcpy.env.cellSize) ** 2) / 1000000 / UnitArea
    
    # Insert rows into the output table
    rasrows = flt_ras.FloatRasterSearchcursor()
    arcpy.AddMessage('factor: %s' % factor)
    with arcpy.da.InsertCursor(Output_Table, ["RASTERVALU", "Area_sqkm"]) as cursor:
        for rasrow in rasrows:
            cursor.insertRow([rasrow.Value, rasrow.Count * factor])
    
    # Calculate the total number of sites
    totalsites = sum(stats_dict.values())
    arcpy.AddMessage('totalsites: %s' % totalsites)
    
    # Initialize variables for cumulative calculations
    totalarea = 0.0
    cumArea = 0
    effarea = []
    nSites = []
    
    # Update the output table with frequency values
    stats_found = 0
    with arcpy.da.UpdateCursor(Output_Table, ["RASTERVALU", "Area_sqkm", "Frequency"]) as cursor:
        for row in cursor:
            tblval = row[0]
            area = row[1]
            totalarea += area
            rasval = flt_ras[tblval]
            if rasval in stats_dict:
                frequency = stats_dict[rasval]
                row[2] = frequency
                cursor.updateRow(row)
                effarea.append(area)
                nSites.append(frequency)
                stats_found += 1
    
    arcpy.AddMessage('stats_found: %s' % stats_found)
    if stats_found < len(stats_dict):
        arcpy.AddError('Not enough Values with Frequency > 0 found!')
        assert False
    elif stats_found > len(stats_dict):
        arcpy.AddError('Too many Values with Frequency > 0 found!')
        assert False
    
    # Calculate cumulative areas and sites
    effarea_rev = reversed(effarea)
    nSites_rev = reversed(nSites)
    effCumarea = 0
    cumSites = 0
    effCumareaList = []
    cumSitesList = []
    for i in range(len(nSites)):
        effCumarea += 100.0 * next(effarea_rev) / totalarea
        effCumareaList.append(effCumarea)
        cumSites += 100.0 * next(nSites_rev) / totalsites
        cumSitesList.append(cumSites)
    
    effCumareaList_rev = reversed(effCumareaList)
    cumSitesList_rev = reversed(cumSitesList)
    
    # Update the output table with cumulative values
    with arcpy.da.UpdateCursor(Output_Table, ["Area_sqkm", "CAPP_CumAr", "Eff_CumAre", "Cum_Sites", "I_CumSites"]) as cursor:
        for row in cursor:
            cumArea += 100.0 * row[0] / totalarea
            row[1] = cumArea
            row[2] = next(effCumareaList_rev)
            Cum_Sites = next(cumSitesList_rev)
            row[3] = Cum_Sites
            row[4] = 100.0 - Cum_Sites
            cursor.updateRow(row)
    
    arcpy.AddMessage('reversed:')
    
    # Calculate efficiency and AUC values
    Eff_CumAre = []
    Cum_Sites = []
    with arcpy.da.SearchCursor(Output_Table, ["Eff_CumAre", "Cum_Sites"]) as cursor:
        next(cursor)  # Skip the first row
        for row in cursor:
            Eff_CumAre.append(row[0])
            Cum_Sites.append(row[1])
    
    sumEff_AUC = 0.0
    with arcpy.da.UpdateCursor(Output_Table, ["Eff_CumAre", "Cum_Sites", "Eff_AUC"]) as cursor:
        for i, row in enumerate(cursor):
            if i < len(Eff_CumAre):
                val = 0.5 * (row[0] - Eff_CumAre[i]) * (row[1] + Cum_Sites[i]) / (100.0 * 100.0)
                sumEff_AUC += val
                row[2] = val
                cursor.updateRow(row)
            else:
                val = 0.5 * (row[0]) * (row[1]) / (100.0 * 100.0)
                sumEff_AUC += val
                row[2] = val
                cursor.updateRow(row)
    
    arcpy.AddMessage('Efficiency: %.1f%%' % (sumEff_AUC * 100.0))
    
    # Remove join if necessary
    if Input_table and joinRastername:
        arcpy.RemoveJoin_management(joinRastername, Input_table)
