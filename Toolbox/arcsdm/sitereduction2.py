# ArcSDM 5 (Arcgis pro)
#Training site reduction  -tool
# 
#
# History:
# Previous version by Unknown (ArcSDM)
# 7.9.2016 Fixes
# 13.4.2016 Recoded for ArcSDM 5 / ArcGis pro
# 30.6.2016 As python toolbox tool module
# 8.8.2016 AG Desktop compatibility TR
# 13.7.2017 Check for non raster masks TR

import sys, string, os, math, traceback
import arcgisscripting
import arcpy
import math

def ReduceSites(self, parameters, messages):
    messages.addMessage("Starting sites reduction")
    messages.addMessage("------------------------------")

    try:
        TrainPts = parameters[0].valueAsText
        OutputPts = parameters[5].valueAsText

        is_thinning_selection_selected = parameters[1].value if parameters[1].value is not None else False
        unit_area_sq_km = parameters[2].value if parameters[2].value is not None else 0.0

        is_random_selection_selected = parameters[3].value if parameters[3].value is not None else False
        percentage = parameters[4].value if parameters[4].value is not None else 100

        if not is_thinning_selection_selected and not is_random_selection_selected:
            gp.AddError("No training sites reduction method selected.")
            raise 'User Error'

        messages.addMessage("%-20s %s" % ("Training points:", TrainPts))

        arcpy.management.SelectLayerByAttribute(TrainPts)

        # Get initial selection within mask
        maskpolygon = arcpy.env.mask
        if maskpolygon is None:
            raise arcpy.ExecuteError("Mask doesn't exist! Set Mask under Analysis/Environments.")
           
        if not arcpy.Exists(arcpy.env.mask):
            raise arcpy.ExecuteError("Mask doesn't exist! Set Mask under Analysis/Environments.")
        maskname = arcpy.Describe(arcpy.env.mask).Name
        
        messages.AddMessage("%-20s %s" % ("Scratch workspace:",  arcpy.env.scratchWorkspace))
        maskpolygon = arcpy.env.mask
        messages.AddMessage("%-20s %s" % ("Using mask:", maskname))
        messages.AddMessage("%-20s %s" % ("Mask type:", arcpy.Describe(arcpy.env.mask).dataType))
        messages.AddMessage("%-20s %s" % ("Selection layer:",OutputPts))
            
        datatype = arcpy.Describe(arcpy.env.mask).dataType
        if (datatype != "FeatureLayer"):
            raise arcpy.ExecuteError("Reduce training points tool requires mask of type 'FeatureLayer!'")

        #This fails with raster mask:
        arcpy.management.MakeFeatureLayer(maskpolygon, maskname)
        #maskname = maskname + "_tmp";
        #arcpy.MakeRasterLayer_management(maskpolygon, maskname)

        arcpy.SelectLayerByLocation_management(TrainPts, 'CONTAINED_BY', maskname, "#", 'SUBSET_SELECTION')
        tpcount = arcpy.GetCount_management(TrainPts)
        #messages.AddMessage("debug: Selected by mask = "+str(tpcount))
        
        thin = parameters[1].valueAsText == 'true'
        #messages.addMessage("debug: thin: = " + str(thin)) ;
        if thin:
            UnitArea = parameters[2].value
        
        #SDMValues.appendSDMValues(gp, UnitArea, TrainPts)
        # This is used to test if OID is OBJECTID or FID
        field_names = [f.name for f in arcpy.ListFields(TrainPts)]
        
        if ('OBJECTID' in field_names):    
            messages.AddMessage("Object contains OBJECTID and is geodatabase feature")
        else:
            messages.AddMessage("Object contains FID and is of type shape")

        if thin:
            #Get minimum allowable distance in meters based on Unit Area
            minDist = math.sqrt(UnitArea * 1000000.0 / math.pi)
            
            #Make list of points from mask-selected featureclass        
            listPnts = []
            feats = arcpy.SearchCursor(TrainPts)
            #messages.addMessage(dir(feats));
            feat = feats.next()
            
            
            while feat:
                pnt = feat.Shape# .GetPart(0)
                # Geodababase
                if ("OBJECTID" in field_names):
                    listPnts.append((pnt,feat.OBJECTID))
                else:
                    listPnts.append((pnt,feat.FID))
                feat = feats.next()
            #gp.AddMessage("%s = %s"%('Num listPnts',listPnts[0]))
                
            #Make list of selected points by making a new list of points
            #not within minimum allowable distance of other points.
            """
                Test point n against list of points saved as having no neighbors within 
                allowable distance of all points.
            """
            NewAlg = 2
            if NewAlg == 2:
                """ Faster processing of table, but same brute-force algorithm as NEwAlg == 1 """
                class POINT(object):
                    def __init__(self, Pnt, FID):
                        """ Pnt is an ESRI geoprocessing point object """
                        self.x = Pnt.X
                        self.y = Pnt.Y
                        self.fid = FID
                    def __eq__(self, otherpnt):
                        return self.x == otherpnt.x and self.y == otherpnt.y
                    def __cmp___(self, otherpnt):
                        if self.x == otherpnt.x and self.y == otherpnt.y: return 0
                        else: return 1
                    def __repr__(self):
                        return "%s, %s, %s"%(self.x, self.y, self.fid)
        
                def rowgen(searchcursor_rows):
                    """ Convert gp searchcursor to a generator function """
                    rows = searchcursor_rows
                    row = rows.next()        
                    while row:
                        yield row
                        row = rows.next()

                def distance(pnt1, pnt0):
                    return math.hypot(pnt1.x - pnt0.x, pnt1.y - pnt0.y)

                def brute_force(savedPnts, unitRadius, point):
                    """
                        1. Add first point to saved list.
                        2. Check if next point is within Unit radius of saved points.
                        3. If not, add it to saved list.
                        4. Go to 2.
                        
                        The number tried is n/2 on average, because saved list grows from 1.
                        nTrials = Sigma(x,x=(1,n)) = n/2 + n*n/4 or O(n*n)
                        nTrials(10) = 5 + 25 = 30
                        nTrials(100) = 50 + 2500 = 2550
                        nTrials(1000) = 500 + 250,000 = 250,500
                        nTrials(10,000) = 5000 + 25,000,000 = 25,050,000
                    """
                    for pnt in savedPnts:
                        d = distance(pnt, point)
                        if d < unitRadius:  return False
                    return True
                #Make list of points from mask-selected featureclass        
                savedPnts = []
                feats = rowgen(arcpy.SearchCursor(TrainPts))
                # This is python 3 specific:
                feat = next(feats)
                if not feat:
                    raise Exception( 'No feature rows selected')
                pnt = feat.Shape.getPart(0)
                if ("OBJECTID" in field_names):
                    point = POINT(pnt, feat.OBJECTID)
                else:
                    point = POINT(pnt, feat.FID)
                
                savedPnts.append(point)

                unitRadius = minDist
                for feat in feats:
                    pnt = feat.Shape.getPart(0)
                    if ("OBJECTID" in field_names):
                        point = POINT(pnt, feat.OBJECTID)
                    else:
                        point = POINT(pnt, feat.FID)
                    if brute_force(savedPnts, unitRadius, point):
                        savedPnts.append(point)
                fidl = savedPnts
                if len(fidl) > 0:
                    #Compose SQL where clause like:  "FID" IN (11, 233, 3010,...)
                    if ("OBJECTID" in field_names): 
                        fids = '"OBJECTID" IN (%d'%fidl[0].fid
                    else:
                        fids = '"FID" IN (%d'%fidl[0].fid
                    for pnt in fidl[1:]:
                        fids += ', %d'%pnt.fid
                    fids += ')'
                    
            elif NewAlg == 1:
                """ This algorithm does not have correct FIDs with points """
                #Make list of points from mask-selected featureclass        
                listPnts = []
                feats = gp.SearchCursor(TrainPts)
                feat = feats.Next()
                while feat:
                    pnt = feat.Shape.GetPart(0)
                    listPnts.append((pnt, feat.FID))
                    feat = feats.Next()
                #gp.AddMessage("%s = %s"%('Num listPnts',listPnts[0]))
                fidl = []
                fidl.append(listPnts[0])
                #gp.AddMessage (str(fidl2))
                for (p0, FID) in listPnts[1:]:
                    OK = 1
                    for (p1, _) in fidl:
                        dst = math.hypot(p1.X-p0.X, p1.Y-p0.Y)
                        #dst = math.sqrt((p1.X-p0.X)**2 + (p1.Y-p0.Y)**2)
                        if dst < minDist:
                            OK = 0
                            break
                    if OK: fidl.append((p0, FID))
                #gp.AddMessage('fidl: %s'%fidl)
                #gp.AddMessage('fidl:'+str(len(fidl))+","+str(fidl))
                #Form selected set from FID list
                fids = 'FID = '
                for (p,fid) in fidl:
                    fids += (fid + ' or FID = ')
                fids = fids[:len(fids)-9] #TODO: Fix this ...
            else:
                ''' Legacy Algorithm
                    This algorithm does not have correct FIDs with points
                '''
                #Make list of points from mask-selected featureclass        
                listPnts = []
                feats = gp.SearchCursor(TrainPts)
                feat = feats.Next()
                while feat:
                    pnt = feat.Shape.GetPart(0)
                    listPnts.append(pnt)
                    feat = feats.Next()
                #gp.AddMessage("%s = %s"%('Num listPnts',listPnts[0]))
                bmSize = tpcount
                fidl = [] #list of indeces of points not close to other points
                first = 1
                s = 1 #number of tests
                fid = 0 #index of tested point and for message at fid mod 100
                for p0 in listPnts:
                    if (fid % 100) == 99: gp.AddMessage("No. tested: %s" % fid)
                    if first:
                        fidl.append(fid)
                        first = 0
                        fid += 1
                        continue
                    else:
                        OK = 1
                        #j is index of tested points.
                        #ith point in points list is tested against jth point in points list
                        #listPnts is ordered on FID
                        for j in range(0,fid):
                            p1 = listPnts[j]
                            s += 1
                            #if jth point index saved in fidl, test ith point against it
                            if j in fidl:
                                dst = math.sqrt((p1.X-p0.X)**2 + (p1.Y-p0.Y)**2)
                                if dst < minDist:
                                    OK = 0
                                    break
                        if OK and (fid<bmSize):# fid is always less than no. of selected points
                            fidl.append(fid)
                    fid += 1
                    s += 1
                    
                #gp.AddMessage('fidl:'+str(len(fidl))+","+str(fidl))
                if ('OBJECTID' in field_names):
                    fids = 'OBJECTID = ('
                else:           
                    fids = 'FID = ('
                for fid in fidl:
                    fids += "%d, "%fid
                fids += ')'
            #gp.AddMessage('fids:'+str(fids))
            total_amount_of_points = (arcpy.GetCount_management(TrainPts))
            
            arcpy.SelectLayerByAttribute_management (TrainPts, 'SUBSET_SELECTION', fids) 
            messages.AddMessage("Selected by thinning = " + str(arcpy.GetCount_management(TrainPts)) + "/" + str(total_amount_of_points )  )

    #Random site reduction can take place after thinning
        if is_random_selection_selected:
            from random import Random
            randomcutoff = float(percentage)
            #Make this validator thing...
            if randomcutoff >= 1 and randomcutoff <= 100:
                randomcutoff = randomcutoff / 100.0
            else:
                messages.AddError("Random cutoff value not in range 1%-100%")
                raise Exception ('User error')
            feats = arcpy.SearchCursor(TrainPts)
            feat = feats.next()
            randnums = []
            import random
            rand = random.Random(None)
            # Notice: We go through the items twice - to make sure we actually pick exabtly the percentage asked - not random
            while feat:
                randnums.append(rand.random())
                feat = feats.next()
            #gp.AddMessage("randnums: " + str(randnums))
            sorted_randnums = randnums * 1
            #gp.AddMessage("sorted_randnums: " + str(sorted_randnums))
            sorted_randnums.sort()
            #messages.addMessage("sorted_randnums: " + str(sorted_randnums))
            #gp.AddMessage("randnums: " + str(randnums))
            cutoff = sorted_randnums[int(randomcutoff * int(arcpy.GetCount_management(TrainPts).getOutput(0)))]
            #gp.AddMessage("cutoff: " + str(cutoff))
            if ('OBJECTID' in field_names):
                fids = 'OBJECTID = '
            else:
                fids = 'FID = '
            feats = arcpy.SearchCursor(TrainPts)
            i = 0
            feat = feats.next()
            #Is this first?
            first = 0
            
            while feat:            
                if randnums[i] < cutoff:
                    if ('OBJECTID' in field_names):
                        if (first > 0):
                            fids += ' or OBJECTID = '
                        else:
                            first = 1;
                        fids += (str(feat.OBJECTID) )
                    else:
                        if (first>0):
                            fids += ' or FID = '
                        else:
                            first = 1
                        fids += (str(feat.fid) )
                i+=1
                feat = feats.next()
            del feats
            #Removing is not done this way... huoh...T
            #fids = fids[:len(fids)-9]
            #messages.AddMessage("Fids: " + fids)
            arcpy.SelectLayerByAttribute_management (TrainPts, 'SUBSET_SELECTION', fids) 
            messages.AddMessage("Selected by random = "+str(arcpy.GetCount_management(TrainPts)))

        if OutputPts: # save as a layer 
            arcpy.CopyFeatures_management(TrainPts, OutputPts)
            arcpy.SetParameterAsText(5, OutputPts)

    except arcpy.ExecuteError as e:
        #TODO: Clean up all these execute errors in final version
        arcpy.AddError("\n");
        arcpy.AddMessage("Training sites reduction caught arcpy.ExecuteError: ")
        args = e.args[0];
        args.split('\n')
        arcpy.AddError(args);
                    
        arcpy.AddMessage("-------------- END EXECUTION ---------------") 
        raise arcpy.ExecuteError;   
    except Exception as Msg:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        #msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
        messages.addErrorMessage(pymsg); #msgs)

        # return gp messages for use with a script tool
        #messages.AddErrorMessage(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        raise
    #gp.MissingDataVariance_sdm(Input_Evidence_Rasters, Output_Weights_Table, crap_pprb, Cellsize)


