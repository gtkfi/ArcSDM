import os
import sys
import math
import arcpy
import numpy as np
from scipy.optimize import minimize

def Execute(self, parameters, messages):
    try:
        # Check out any necessary licenses
        arcpy.CheckOutExtension("spatial")

        # Set environment settings
        arcpy.env.overwriteOutput = True
        arcpy.env.addOutputsToMap = True
        arcpy.env.scratchWorkspace = parameters[11].valueAsText

        # Get parameters
        unitCell = parameters[5].value
        if unitCell < (float(arcpy.env.cellSize) / 1000.0) ** 2:
            unitCell = (float(arcpy.env.cellSize) / 1000.0) ** 2
            arcpy.AddWarning('Unit Cell area is less than area of Study Area cells.\n' +
                             'Setting Unit Cell to area of study area cells: %.0f sq km.' % unitCell)

        # Get evidence layer names
        Input_Rasters = parameters[0].valueAsText.split(';')
        for i, s in enumerate(Input_Rasters):
            Input_Rasters[i] = s.strip("'")

        # Get evidence layer types
        Evidence_types = parameters[1].valueAsText.lower().split(';')
        if len(Evidence_types) != len(Input_Rasters):
            arcpy.AddError("Not enough Evidence types!")
            raise Exception

        # Get weights tables names
        Wts_Tables = parameters[2].valueAsText.split(';')
        if len(Wts_Tables) != len(Input_Rasters):
            arcpy.AddError("Not enough weights tables!")
            raise Exception

        # Get Training sites feature layer
        TrainPts = parameters[3].valueAsText
        
        # Get Missing data value
        missing_data_value = parameters[4].valueAsText

        # Get output raster name
        thmUC = arcpy.CreateScratchName("tmp_UCras", '', 'raster', arcpy.env.scratchWorkspace)
        
        ot = [['%s, %s'%(Input_Rasters[i], Wts_Tables[i])] for i in range(len(Input_Rasters))]
        # Create Generalized Class tables
        Wts_Rasters = []
        for Input_Raster, Wts_Table in zip(Input_Rasters, Wts_Tables):
            Output_Raster = arcpy.CreateScratchName(arcpy.Describe(Input_Raster).catalogPath[:9] + '_G', '', 'raster', arcpy.env.scratchWorkspace)

            # If using GDB database, remove numbers and underscore from the beginning of the Output_Raster #AL 061020
            outbase = os.path.basename(Output_Raster)
            if arcpy.Describe(arcpy.env.workspace).workspaceType != "FileSystem":
                outbase = os.path.basename(Output_Raster)
            while len(outbase) > 0 and (outbase[:1] <= "9" or outbase[:1] == "_"):
                outbase = outbase[1:]
            Output_Raster = os.path.dirname(Output_Raster) + "\\" + outbase

            arcpy.env.snapRaster = Input_Raster

            # Specify the fields to join on
            join_field_fc = "Value"  # Field in the feature class
            join_field_table = "CLASS"  # Field in the table
            # Create a new field in the Input_Raster to store the joined values
            joined_field = "GEN_CLASS"
            if not arcpy.ListFields(Input_Raster, joined_field):
                arcpy.management.AddField(Input_Raster, joined_field)

            # Create a dictionary to store the join values from Wts_Table
            join_values = {}
            with arcpy.da.SearchCursor(Wts_Table, [join_field_table, joined_field]) as cursor:
                for row in cursor:
                    join_values[row[0]] = row[1]

            # Update the Input_Raster with the joined values
            with arcpy.da.UpdateCursor(Input_Raster, [join_field_fc, joined_field]) as cursor:
                for row in cursor:
                    if row[0] in join_values:
                        row[1] = join_values[row[0]]
                    else:
                        row[1] = None
                    cursor.updateRow(row)

            arcpy.sa.Lookup(Input_Raster, "GEN_CLASS").save(Output_Raster)

            if not arcpy.Exists(Output_Raster):
                arcpy.AddError(Output_Raster + " does not exist.")
                raise Exception
            Wts_Rasters.append(arcpy.Describe(Output_Raster).catalogPath)

        for wts_raster in Wts_Rasters:
            arcpy.management.BuildRasterAttributeTable(wts_raster)

        # Create the Unique Conditions raster from Generalized Class rasters
        thmUC = arcpy.sa.Combine(Wts_Rasters)

        # Get UC lists from combined raster
        UCOIDname = arcpy.Describe(thmUC).OIDFieldName
        evflds = [fld.name for fld in arcpy.ListFields(thmUC) if fld.name != UCOIDname and fld.name.upper() not in ('VALUE', 'COUNT')]
        lstsVals = [[] for fld in evflds]
        cellSize = float(arcpy.env.cellSize)
        lstAreas = [[] for fld in evflds]

        thmUCRL = arcpy.MakeRasterLayer_management(thmUC, "thmUCRL").getOutput(0)
        with arcpy.da.SearchCursor(thmUCRL, evflds + ['COUNT']) as ucrows:
            for ucrow in ucrows:
                for i, fld in enumerate(evflds):
                    lstsVals[i].append(ucrow[i])
                    lstAreas[i].append(ucrow[-1] * cellSize * cellSize / (1000000.0 * unitCell))

        # Handle missing values by replacing them with a weighted mean
        for i in range(len(lstsVals)):
            values = np.array(lstsVals[i])
            weights = np.array(lstAreas[i])
            mask = values == float(missing_data_value)
            weighted_mean = np.average(values[~mask], weights=weights[~mask])
            values[mask] = weighted_mean
            lstsVals[i] = values.tolist()

        # Check Maximum area of conditions so not to exceed 100,000 unit areas
        maxArea = max(lstAreas[0])
        if (maxArea / unitCell) / 100000.0 > 1:
            unitCell = math.ceil(maxArea / 100000.0)
            arcpy.AddWarning('UnitCell is set to minimum %.0f sq. km. to avoid area limit in Logistic Regression!' % unitCell)

        # Get Number of Training Sites per UC Value
        siteFIDName = 'TPFID'
        if siteFIDName not in [field.name for field in arcpy.ListFields(TrainPts)]:
            arcpy.management.AddField(TrainPts, siteFIDName, 'LONG')

        TrainingPtsOID = arcpy.Describe(TrainPts).OIDFieldName
        arcpy.CalculateField_management(TrainPts, siteFIDName, "!{}!".format(TrainingPtsOID))

        tempExtrShp = arcpy.CreateScratchName('Extr', 'Tmp', 'shapefile', arcpy.env.scratchWorkspace)
        arcpy.sa.ExtractValuesToPoints(TrainPts, thmUC, tempExtrShp)

        # Make dictionary of Counts of Points per RasterValue
        CntsPerRasValu = {}
        with arcpy.da.SearchCursor(tempExtrShp, ['RasterValu']) as tpFeats:
            for tpFeat in tpFeats:
                if tpFeat[0] in CntsPerRasValu.keys():
                    CntsPerRasValu[tpFeat[0]] += 1
                else:
                    CntsPerRasValu[tpFeat[0]] = 1

        # Make Number of Points list in RasterValue order
        lstPnts = []
        numUC = len(lstsVals[0])
        for i in range(1, numUC + 1):  # Combined raster values start at 1
            if i in CntsPerRasValu.keys():
                lstPnts.append(CntsPerRasValu.get(i))
            else:
                lstPnts.append(0)

        # Perform logistic regression using numpy for multi-class support
        case_data = np.array([lstsVals[0], lstPnts, lstAreas[0]]).T
        X = case_data[:, :-2]  # Features
        y = case_data[:, -2].astype(int)  # Target variable (number of points)
        # Number of classes
        num_classes = len(np.unique(y))
        y = np.clip(y, 0, num_classes - 1)  # Ensure y values are within the range of num_classes
        sample_weight = case_data[:, -1]  # Sample weights (area)

        # Define the softmax function
        def softmax(z):
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Define the cost function for multi-class logistic regression
        def cost_function(params, X, y, sample_weight, num_classes):
            m, n = X.shape
            params = params.reshape((num_classes, n + 1))
            X_with_intercept = np.hstack([np.ones((m, 1)), X])
            logits = np.dot(X_with_intercept, params.T)
            probs = softmax(logits)
            y_one_hot = np.eye(num_classes)[y.astype(int)]
            cost = -np.sum(sample_weight * y_one_hot * np.log(probs)) / m
            return cost

        # Define the gradient function for multi-class logistic regression
        def gradient(params, X, y, sample_weight, num_classes):
            m, n = X.shape
            params = params.reshape((num_classes, n + 1))
            X_with_intercept = np.hstack([np.ones((m, 1)), X])
            logits = np.dot(X_with_intercept, params.T)
            probs = softmax(logits)
            y_one_hot = np.eye(num_classes)[y.astype(int)]
            grad = np.dot((probs - y_one_hot).T, X_with_intercept) / m
            grad = grad * sample_weight[:, np.newaxis]
            return grad.flatten()

        # Initial parameters (including intercept)
        initial_params = np.zeros((num_classes, X.shape[1] + 1)).flatten()

        # Minimize the cost function
        result = minimize(cost_function, initial_params, args=(X, y, sample_weight, num_classes), method='BFGS', jac=gradient)
        params = result.x.reshape((num_classes, X.shape[1] + 1))

        intercept = params[:, 0]
        print("Intercept:", intercept)  # Access the intercept to avoid the compile error
        intercept = params[:, 0]
        coefficients = params[:, 1:]

        # Define lstLine variable with logistic regression results
        lstLine = []
        for i in range(num_classes):
            lstLine.append([
                str(i + 1),  # ID
                str(params[i, 0]),  # LRPostProb
                str(params[i, 1]) if params.shape[1] > 1 else '0',  # LR_Std_Dev
                str(params[i, 2]) if params.shape[1] > 2 else '0'   # LRTValue
            ])

        # Create a table to hold logistic regression results
        fnNew = parameters[6].valueAsText
        tblbn = os.path.basename(fnNew)
        [tbldir, tblfn] = os.path.split(fnNew)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew = fnNew[:-4] if fnNew.endswith(".dbf") else fnNew
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        arcpy.AddMessage("fnNew: %s" % fnNew)
        arcpy.AddMessage('Making table to hold logistic regression results (param 6): %s' % fnNew)
        fnNew = tblbn
        arcpy.management.CreateTable(tbldir, tblfn)
        arcpy.management.AddField(fnNew, 'ID', 'LONG', 6)
        arcpy.management.AddField(fnNew, 'LRPostProb', 'Double', "#", "#", "#", "LR_Posterior_Probability")
        arcpy.management.AddField(fnNew, 'LR_Std_Dev', 'Double', "#", "#", "#", "LR_Standard_Deviation")
        arcpy.management.AddField(fnNew, 'LRTValue', 'Double', "#", "#", "#", "LR_TValue")
        arcpy.management.DeleteField(fnNew, "Field1")
        vTabLR = fnNew

        # Insert logistic regression results into the table
        with arcpy.da.InsertCursor(vTabLR, ["ID", "LRPostProb", "LR_Std_Dev", "LRTValue"]) as cursor:
            for line in lstLine:
                cursor.insertRow(line)
        arcpy.AddMessage('Created table to hold logistic regression results: %s' % fnNew)

        lstLabels = []
        for el in ot:
            for e in el:
                lstLabels.append(e.replace(' ', ''))
        # Create a table to hold theme coefficients
        fnNew2 = parameters[7].valueAsText
        tblbn = os.path.basename(fnNew2)
        [tbldir, tblfn] = os.path.split(fnNew2)
        if tbldir.endswith(".gdb"):
            tblfn = tblfn[:-4] if tblfn.endswith(".dbf") else tblfn
            fnNew2 = fnNew2[:-4] if fnNew2.endswith(".dbf") else fnNew2
            tblbn = tblbn[:-4] if tblbn.endswith(".dbf") else tblbn
        arcpy.AddMessage("Making table to hold theme coefficients: " + fnNew2)
        arcpy.management.CreateTable(tbldir, tblfn)
        arcpy.management.AddField(fnNew2, "Theme_ID", 'Long', 6, "#", "#", "Theme_ID")
        arcpy.management.AddField(fnNew2, "Theme", 'text', "#", "#", 256, "Evidential_Theme")
        arcpy.management.AddField(fnNew2, "Coeff", 'double', "#", "#", "#", 'Coefficient')
        arcpy.management.AddField(fnNew2, "LR_Std_Dev", 'double', "#", "#", "#", "LR_Standard_Deviation")
        arcpy.management.DeleteField(fnNew2, "Field1")
        vTabLR2 = fnNew2

        # Insert theme coefficients into the table
        with arcpy.da.InsertCursor(vTabLR2, ["Theme_ID", "Theme", "Coeff", "LR_Std_Dev"]) as cursor:
            for i, coef in enumerate(coefficients):
                cursor.insertRow((i + 1, "Theme " + str(i + 1), coef[0], 0))  # Placeholder for standard deviation
        arcpy.AddMessage('Created table to hold theme coefficients: %s' % fnNew2)

        # Creating LR Response Rasters
        arcpy.AddMessage("Creating LR Response Rasters")
        # Join LR polynomial table to unique conditions raster and copy to get a raster with attributes
        cmb = thmUC
        tbl = parameters[6].valueAsText
        tbltv = 'tbltv'
        arcpy.management.MakeTableView(tbl, tbltv)

        # Perform the join
        arcpy.management.JoinField(cmb, 'VALUE', tbl, 'ID')

        # Make output float rasters from attributes of joined unique conditions raster
        outRaster1 = parameters[8].valueAsText
        outRaster2 = parameters[9].valueAsText
        outRaster3 = parameters[10].valueAsText

        outcon1 = arcpy.sa.Con(cmb, arcpy.sa.Lookup(cmb, "LRPostProb"), 0, "LRPostProb > 0")
        outcon1.save(outRaster1)
        outcon2 = arcpy.sa.Con(cmb, arcpy.sa.Lookup(cmb, "LR_Std_Dev"), 0, "LR_Std_Dev > 0")
        outcon2.save(outRaster2)
        outcon3 = arcpy.sa.Con(cmb, arcpy.sa.Lookup(cmb, "LRTValue"), 0, "LRTValue > 0")
        outcon3.save(outRaster3)

        # Add to display
        arcpy.SetParameterAsText(6, tbl)
        arcpy.SetParameterAsText(7, arcpy.Describe(vTabLR2).catalogPath)
        arcpy.SetParameterAsText(8, outRaster1)
        arcpy.SetParameterAsText(9, outRaster2)
        arcpy.SetParameterAsText(10, outRaster3)

    except arcpy.ExecuteError as e:
        arcpy.AddError("\n")
        arcpy.AddMessage("Caught ExecuteError in logistic regression. Details:")
        args = e.args[0]
        args.split('\n')
        arcpy.AddError(args)
    except:
        pymsg = "PYTHON ERRORS: \n Error Info:\n    " + msgs + "\n"
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)
        arcpy.AddError(pymsg)
        print(pymsg)
        raise