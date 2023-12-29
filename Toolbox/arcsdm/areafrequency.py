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

class UserException(Exception):
    pass

import sys, string, os, random, traceback, tempfile
import arcgisscripting
import arcpy
import arcsdm.workarounds_93
from arcsdm.floatingrasterclass import FloatRasterVAT, rowgen
import importlib;

import arcpy;
# Create the Geoprocessor object
gp = arcgisscripting.create()

# Check out any necessary licenses
gp.CheckOutExtension("spatial")

gp.OverwriteOutput = 1

# Script arguments...
def Execute(self, parameters, messages):
    import arcsdm.sdmvalues
    try:
        importlib.reload (arcsdm.sdmvalues)
    except:
        reload(arcsdm.sdmvalues);
    # try:
    Input_point_features = parameters[0].valueAsText
    Input_raster =  parameters[1].valueAsText # gp.GetParameterAsText(1)
    Value_field =  parameters[2].valueAsText # gp.GetParameterAsText(2)
    UnitArea =  parameters[3].value # gp.GetParameter(3)
    Output_Table =  parameters[4].valueAsText # gp.GetParameterAsText(4)
    arcsdm.sdmvalues.appendSDMValues(gp, UnitArea, Input_point_features)
    arcpy.AddMessage("\n"+"="*10+" Starting area frequency "+"="*10)
    
    # Some locals
    valuetypes = {1:'Integer', 2:'Float'}
    joinRastername = None
    Input_table = None
    RasterValue_field = Value_field.title().endswith('Value')
    # if (RasterValue_field):
    #    arcpy.AddMessage("Debug: There is rastervaluefield");
    # else:
    #    arcpy.AddMessage("Debug: There is no rastervaluefield");
    
    # Create Output Raster
    valuetype = gp.GetRasterProperties (Input_raster, 'VALUETYPE')
    gp.addmessage("Valuetype = " + str(valuetype))
    # gp.addmessage("Value type: %s"%valuetypes[valuetype])
    # if valuetypes[valuetype].title() == 'Integer': #INTEGER #RDB
    if valuetype <= 8:  #<RDB new integer valuetype property values for arcgis version 10
        if not Input_table:
            if not RasterValue_field:
                # Create a float raster from a floating attribute field
                float_type = ('Double','Single')
                fld = gp.listfields(Input_raster, Value_field).__next__()
                Value_field_type = fld.type
                if Value_field_type.title() in float_type:
                    Value_field = Value_field.split('.')
                    if len(Value_field) > 1:
                        gp.adderror("Integer Raster has joined table.")
                        # raise RunTimeError("Integer Raster has joined table.")                         
                        raise UserException
                    InExpression = "FLOAT(%s.%s)"%(Input_raster, Value_field[0])
                    # gp.addwarning(InExpression)
                    TmpRaster = gp.createscratchname("tmp_AFT_ras", "", "raster", gp.scratchworkspace)
                    gp.SingleOutputMapAlgebra_sa(InExpression, TmpRaster)
                    gp.addmessage("Floating Raster from Raster Attribute: type %s"%gp.describe(Input_raster).pixeltype)
                else:
                    gp.adderror("Integer Raster Attribute field not floating type.")
                    # raise RunTimeError("Integer Raster Attribute field not floating type.")
                    raise UserException
            else:
                # Create a float raster from the Value field
                gp.adderror("Integer Raster Value field not acceptable.")
                # raise
                # raise RunTimeError ("Integer Raster Value field not acceptable")
                # raise arcpy.ExecuteError ("Integer Raster Value field not acceptable")
                raise UserException ("Integer Raster Value field not acceptable")
                
        Input_raster = TmpRaster # The input raster is now the new float raster
        valuetype = 2 # Always a floating point raster
    else: # FLOAT
        gp.addmessage("Floating Raster from Floating Raster Value: type %s"%gp.describe(Input_raster).pixeltype)
        
    # Process: Extract Values of Input Raster to Training Points...
    # gp.AddMessage("tpcnt = %i"%gp.GetCount_management(Input_point_features))
    if gp.GetCount_management(Input_point_features) == 0:
        gp.AddError("Training Points must be selected: %s"%Input_point_features)
        raise UserException
    # gp.AddMessage('Extracting values to points...')
    # Output_point_features = gp.createuniquename("Extract_Train.shp", gp.ScratchWorkspace)
    # gp.ExtractValuesToPoints_sa(Input_point_features, Input_raster, Output_point_features)
    Output_point_features = arcsdm.workarounds_93. ExtractValuesToPoints(gp, Input_raster, Input_point_features, "TPFID")
    
# Process: Summary Statistics...
    # Get stats of RASTERVALU field in training sites features with extracted points.
    # gp.AddMessage('Getting statistics...')
    
    # TODO: IF GDB, no .dbf if other - .dbf        
    # Output_summary_stats = gp.createuniquename("Ext_Trn_Stats.dbf", gp.scratchworkspace)
    Output_summary_stats = gp.createuniquename("Ext_Trn_Stats", gp.scratchworkspace)
    
    

    
    stats_dict = {}
    # gp.addwarning('Got stats...')
    # Get all VALUES from input raster, add to stats_dict dictionary
    # from floatingrasterclass import FloatRasterVAT, rowgen
    flt_ras = FloatRasterVAT(gp, Input_raster)
    
    rows = flt_ras.FloatRasterSearchcursor()
    gp.Statistics_analysis(Output_point_features, Output_summary_stats,"RASTERVALU FIRST","RASTERVALU")
    
    
    
    for row in rows: stats_dict[row.value] = 0
    num_training_sites = gp.getcount(Output_point_features)
    # gp.addwarning('num_training_sites: %s'%num_training_sites)
    # Get frequency of RASTERVALU in training points extracted values.
    statsrows = rowgen(gp.SearchCursor(Output_summary_stats))
    
    
    
    
    num_nodata = 0
    for row in statsrows:
        
        # Get actual raster value from truncated value in Extracted values of point theme.
        # gp.addwarning( 'row.RASTERVALU: %s'%row.RASTERVALU)
        # gp.addmessage( list(flt_ras.vat))
        # gp.addwarning( flt_ras[row.RASTERVALU])
        if row.RASTERVALU == flt_ras.getNODATA():
            num_nodata = row.FREQUENCY
        else:
            # NODATA value not included in table
            rasval = flt_ras[row.RASTERVALU]
            # Update stats dictionary with occurence frequencies in Statistics table
            if rasval in stats_dict: stats_dict[rasval] = row.FREQUENCY
    # gp.addwarning("Created stats_dict: %s"%stats_dict)
    
    num_counts = sum(stats_dict.values())
    if num_counts != num_training_sites - num_nodata:
        gp.addwarning("Stats count and number of training sites in data area do not compare.")
    if num_nodata > 0: gp.addwarning("%d training points in NoData area."%num_nodata)
    # gp.AddMessage(Output_summary_stats);
    # raise                                                                        
    gp.AddMessage('Creating table: %s'%Output_Table)
    fullname = arcpy.ParseTableName(Output_Table);
    database, owner, table = fullname.split(", ")
    gp.AddMessage('Output workspace: %s'%os.path.dirname(Output_Table))
    
    gp.AddMessage('Output table name: %s'%os.path.basename(Output_Table))
    gp.CreateTable_management(os.path.dirname(Output_Table),os.path.basename(Output_Table))
    # gp.AddMessage("Created output table.")
    gp.MakeTableView(Output_Table, 'output_table')
    
    gp.AddField_management('output_table', "Frequency", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    # Precision and scale of RASTERVALU field must be same as field of that name in extract and statistics tables
    gp.AddField_management('output_table', "RASTERVALU", "DOUBLE", "18", "8", "", "", "NULLABLE", "NON_REQUIRED", "")
    
    # Process: Add Field (2)...

    # Process: Add Field (3)...
    gp.AddField_management('output_table', "Area_sqkm", "DOUBLE", "", "", "", "Area_Sq_Kilometers", "NULLABLE", "NON_REQUIRED", "")

    # Process: Add Field (5)...
    gp.AddField_management('output_table', "CAPP_CumAr", "DOUBLE", "", "", "", "CAPP_Cumulative_Area", "NULLABLE", "NON_REQUIRED", "")
    #gp.AddMessage("got to here....9")

    # Process: Add Field (6)...
    gp.AddField_management('output_table', "Eff_CumAre", "DOUBLE", "", "", "", "Efficiency_Cumulative_Area", "NULLABLE", "NON_REQUIRED", "")
    # gp.AddMessage("got to here....10")

    # Process: Add Field (7)...
    gp.AddField_management('output_table', "Cum_Sites", "DOUBLE", "", "", "", "Cumulative_Sites", "NULLABLE", "NON_REQUIRED", "")

    # Process: Add Field (7)...
    gp.AddField_management('output_table', "I_CumSites", "DOUBLE", "", "", "", "Cumulative_Sites", "NULLABLE", "NON_REQUIRED", "")

    # Process: Add Field (7)...
    gp.AddField_management('output_table', "Eff_AUC", "DOUBLE", "", "", "", "A_U_C", "NULLABLE", "NON_REQUIRED", "")

    # gp.AddMessage("Created output table and added fields.")

    gp.DeleteField_management(Output_Table, "Field1")
    
# Calculate Count, Area, and Percent fields
    # gp.AddMessage("got to here....11")
    # gp.AddWarning('Assume cell size units is meters!')
    factor = (float(gp.CellSize)**2) / 1000000 / UnitArea
    # gp.AddMessage(str(factor))
    # gp.AddMessage("Input_raster path=%s"%os.path.basename(Input_raster))

# Search raster must be a raster layer
    # gp.addmessage("Value type: %s"%valuetypes[valuetype])
    rasrows = flt_ras.FloatRasterSearchcursor()
    # gp.addmessage('Opened Float Raster Searchcursor...')

# Insert some field values in output table
    # Open insert cursor for output
    tblrows = gp.InsertCursor(Output_Table)
    gp.AddMessage('factor: %s'%factor)
    # Create Output records
    for rasrow in rasrows:
        tblrow = tblrows.NewRow()
        tblrow.RASTERVALU = rasrow.Value
        # gp.addwarning(str(rasrow.Value))
        tblrow.Area_sqkm = rasrow.Count * factor
        tblrows.InsertRow(tblrow)

    del tblrow,tblrows
    # gp.AddMessage("No. records in output table %s: %i"%(Output_Table,gp.GetCount_management(Output_Table)))

    # Get total sites from stats table
    totalsites = sum(stats_dict.values())
    gp.AddMessage('totalsites: %s'%totalsites)
    # Variables for more stuff
    totalarea = 0.0
    cumArea = 0
    effarea = []
    nSites = []
    
    # Update Frequency field and get two summations and create two lists
    # gp.addmessage('statvals: '+str(stats_dict.keys()))
    # gp.AddMessage("Calculating Frequency field...")
    tblrows = gp.UpdateCursor(Output_Table)
    tblrow = tblrows.Next()
    stats_found = 0
    while tblrow:
        tblval = tblrow.RASTERVALU
        area = tblrow.Area_sqkm
        totalarea += area
        # tblval is less precision than rasval
        rasval = flt_ras[tblval]
        if rasval in stats_dict:
            frequency = stats_dict[rasval]
            # gp.AddMessage("Found tblval = %s; frequency = %s"%(tblval,frequency))
            tblrow.Frequency = frequency
            tblrows.UpdateRow(tblrow)
            effarea.append(area)
            nSites.append(frequency)
            stats_found += 1
            # gp.AddMessage("Debug: Stats_found =  %s"%stats_found);
        
        tblrow = tblrows.Next()
    
    del tblrow,tblrows
    gp.AddMessage('stats_found: %s'%stats_found)
    # Check that output table is consistent with statistics
    if stats_found < len(stats_dict):
        gp.adderror('Not enough Values with Frequency > 0 found!')
        assert False
    elif stats_found > len(stats_dict):
        gp.adderror('Too many Values with Frequency > 0 found!')
        assert False
    else:
        pass
        # gp.addmessage('All Values with Frequency > 0 found')

    # From two reversed lists, create two lists and two cumulative summations
    # gp.AddMessage("Calculating CAPP_CumAre,Eff_CumAre,Cum_Sites,I_CumSites fields...")
    effarea_rev = reversed(effarea) # generator
    nSites_rev = reversed(nSites) # generator
    effCumarea=0
    cumSites=0
    effCumareaList = []
    cumSitesList = [] 
    for i in range(len(nSites)):
        effCumarea += 100.0 * next(effarea_rev) / totalarea
        effCumareaList.append(effCumarea)
        cumSites += 100.0 * next(nSites_rev) / totalsites
        cumSitesList.append(cumSites)
        
    # Update four fields from reversed lists
    effCumareaList_rev = reversed(effCumareaList) # generator
    cumSitesList_rev = reversed(cumSitesList) # generator 
    # gp.AddMessage('doing update....')
    tblrows = gp.UpdateCursor(Output_Table)
    # gp.AddMessage(str(tblrows))
    tblrow = tblrows.Next()
    while tblrow:
        # gp.AddMessage(str(tblrow) + str(i))
        cumArea += 100.0 * tblrow.Area_sqkm / totalarea
        tblrow.CAPP_CumAr = cumArea
        tblrow.Eff_CumAre = next(effCumareaList_rev)
        Cum_Sites = next(cumSitesList_rev)
        tblrow.Cum_Sites = Cum_Sites
        tblrow.SetValue('I_CumSites', 100.0 - Cum_Sites)
        tblrows.UpdateRow(tblrow)
        tblrow = tblrows.Next()
    # gp.addmessage('done.')
    del tblrow, tblrows
    gp.AddMessage('reversed:')
    # Create two more lists
    # gp.AddMessage("Calculating Eff_AUC field...")
    Eff_CumAre = []
    Cum_Sites = []
    tblrows2 = gp.SearchCursor(Output_Table)
    tblrow2 = tblrows2.Next()
    tblrow2 = tblrows2.Next()
    while tblrow2:
        Eff_CumAre.append(tblrow2.Eff_CumAre)
        Cum_Sites.append(tblrow2.Cum_Sites)
        tblrow2 = tblrows2.Next()

    # Finally, calculate the Eff_AUC field from two lists and Efficiency
    sumEff_AUC = 0.0
    tblrows1 = gp.UpdateCursor(Output_Table)
    tblrow1 = tblrows1.Next()
    for i in range(len(Eff_CumAre)):
        val = 0.5 * (tblrow1.Eff_CumAre - Eff_CumAre[i]) * (tblrow1.Cum_Sites + Cum_Sites[i]) / (100.0 * 100.0)
        sumEff_AUC += val
        # gp.AddMessage(str(val))
        tblrow1.Eff_AUC = val
        tblrows1.UpdateRow(tblrow1)
        tblrow1 = tblrows1.Next()
    # Calculate last row
    if tblrow1:
        # gp.AddMessage("Calculating last row...")
        val = 0.5 * (tblrow1.Eff_CumAre) * (tblrow1.Cum_Sites) / (100.0 * 100.0)
        sumEff_AUC += val
        tblrow1.Eff_AUC = val
        # gp.AddMessage(str(val))
        tblrows1.UpdateRow(tblrow1)
    del tblrow1,tblrows1

    gp.addmessage('Efficiency: %.1f%%'%(sumEff_AUC*100.0))

    if Input_table and joinRastername: # In case of joined integer raster and table
        gp.RemoveJoin_management(joinRastername, Input_table)
            
            
    # except UserException:
    #     print('User exception caught. ')
        
    # except arcpy.ExecuteError:
    #     #TODO: Clean up all these execute errors in final version
    #     gp.AddMessage("AreaFrequency caught: arcpy.ExecuteError");
    #     gp.AddMessage("-------------- END EXECUTION ---------------");        
    #     raise 
    # except:
    #     #In case of joined integer raster and table
    #     arcpy.AddMessage("Tsip");
    #     if Input_table and joinRastername:
    #         gp.RemoveJoin_management(joinRastername, Input_table)
    #    # get the traceback object
    #     tb = sys.exc_info()[2]
    #     # tbinfo contains the line number that the code failed on and the code from that line
    #     tbinfo = traceback.format_tb(tb)[0]
    #     # concatenate information together concerning the error into a message string
    #     pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
    #         str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
    #     # generate a message string for any geoprocessing tool errors
    #     msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
    #     gp.AddError(msgs)

    #     # return gp messages for use with a script tool
    #     gp.AddError(pymsg)

    #     # print messages for use in Python/PythonWin
    #     print (pymsg)
    #     print (msgs)
    #     raise
