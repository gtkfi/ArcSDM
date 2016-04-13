# ArcSDM 5 (Arcgis pro)
#Training site reduction  -tool
# 
#
# History:
# Previous version by Unknown (ArcSDM)
# 13.4.2016 Recoded for ArcSDM 5 / ArcGis pro
#
# TODO: Cleanup 
#  make input optionally points and result another table? (is not commandline testable now)


# Import system modules
import sys, string, os, math, traceback
import SDMValues
import arcgisscripting

gp = arcgisscripting.create()

# Check out any necessary licenses
gp.CheckOutExtension("spatial")

# Load required toolboxes...
#d = gp.GetInstallInfo()
#sPath = d["InstallDir"]
##gp.AddToolbox("C:/Program Files/ArcGIS/ArcToolbox/Toolboxes/Spatial Analyst Tools.tbx") #<== RDB 07/01/2010
##gp.AddToolbox("C:/Program Files/ArcGIS/ArcToolbox/Toolboxes/Data Management Tools.tbx")
#gp.AddToolbox(os.path.join(sPath, "ArcToolbox/Toolboxes/Spatial Analyst Tools.tbx"))
#gp.AddToolbox(os.path.join(sPath, "ArcToolbox/Toolboxes/Data Management Tools.tbx"))

# Script arguments...
# Local variables...

# Process: Missing Data Variance...
try:
    TrainPts = gp.GetParameterAsText(0)
    gp.SelectLayerByAttribute_management (TrainPts) 
    #gp.AddMessage("%s All Selected = %s"%(TrainPts,str(gp.GetCount_management(TrainPts))))
    
    #Get initial selection within mask
    maskname = gp.Describe(gp.mask).Name+'_poly'
    maskpolygon = os.path.join(gp.ScratchWorkspace,maskname)
    if not gp.Exists(maskpolygon):
        gp.RasterToPolygon_conversion(gp.mask, maskpolygon, "SIMPLIFY")
    gp.MakeFeatureLayer_management(maskpolygon, maskname)
    gp.SelectLayerByLocation(TrainPts, 'CONTAINED_BY', maskname, "#", 'SUBSET_SELECTION')
    tpcount = gp.GetCount_management(TrainPts)
    #gp.AddMessage("Selected by mask = "+str(tpcount))
    
    thin = gp.GetParameterAsText(1) == 'true'
    UnitArea = float(gp.GetParameterAsText(2))
    random = gp.GetParameterAsText(3) == 'true'
    
    SDMValues.appendSDMValues(gp, UnitArea, TrainPts)
    
    if thin:
        #Get minimum allowable distance in meters based on Unit Area
        minDist = math.sqrt(UnitArea * 1000000.0 / math.pi)
        
        #Make list of points from mask-selected featureclass        
        listPnts = []
        feats = gp.SearchCursor(TrainPts)
        feat = feats.Next()
        while feat:
            pnt = feat.Shape.GetPart(0)
            listPnts.append((pnt,feat.FID))
            feat = feats.Next()
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
            feats = rowgen(gp.SearchCursor(TrainPts))
            # This is python 3 specific:
            feat = next(feats)
            if not feat:
                raise Exception( 'No feature rows selected')
            pnt = feat.Shape.GetPart(0)
            point = POINT(pnt, feat.FID)
            savedPnts.append(point)

            unitRadius = minDist
            for feat in feats:
                pnt = feat.Shape.GetPart(0)
                point = POINT(pnt, feat.FID)
                if brute_force(savedPnts, unitRadius, point):
                    savedPnts.append(point)
            fidl = savedPnts
            if len(fidl) > 0:
                #Compose SQL where clause like:  "FID" IN (11, 233, 3010,...)
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
            fids = fids[:len(fids)-9]
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
            fids = 'FID = ('
            for fid in fidl:
                fids += "%d, "%fid
            fids += ')'
        #gp.AddMessage('fids:'+str(fids))
                        
        gp.SelectLayerByAttribute_management (TrainPts, 'SUBSET_SELECTION', fids) 
        gp.AddMessage("Selected by thinning = " + str(gp.GetCount_management(TrainPts)))

#Random site reduction can take place after thinning
    if random:
        from random import Random
        randomcutoff = float(gp.GetParameterAsText(4))
        if randomcutoff >= 1 and randomcutoff <= 100:
            randomcutoff = randomcutoff / 100.0
        else:
            gp.AddError("Random cutoff value not in range 1%-100%")
            raise Exception ('User error')
        feats = gp.SearchCursor(TrainPts)
        feat = feats.Next()
        randnums = []
        import random
        rand = random.Random(None)
        while feat:
            randnums.append(rand.random())
            feat = feats.Next()
        #gp.AddMessage("randnums: " + str(randnums))
        sorted_randnums = randnums * 1
        #gp.AddMessage("sorted_randnums: " + str(sorted_randnums))
        sorted_randnums.sort()
        #gp.AddMessage("sorted_randnums: " + str(sorted_randnums))
        #gp.AddMessage("randnums: " + str(randnums))
        cutoff = sorted_randnums[int(randomcutoff * gp.GetCount_management(TrainPts))]
        #gp.AddMessage("cutoff: " + str(cutoff))
        fids = 'FID = '
        feats = gp.SearchCursor(TrainPts)
        i = 0
        feat = feats.Next()
        while feat:
            if randnums[i] < cutoff:
                fids += (str(feat.fid) + ' or FID = ')
            i+=1
            feat = feats.Next()
        del feats
        fids = fids[:len(fids)-9]
        #gp.AddMessage(fids)
        gp.SelectLayerByAttribute_management (TrainPts, 'SUBSET_SELECTION', fids) 
        gp.AddMessage("Selected by random = "+str(gp.GetCount_management(TrainPts)))
    if not thin and not random:
        gp.AddError("No training sites reduction method selected.")
        raise 'User Error'

except Exception as Msg:
    # get the traceback object
    tb = sys.exc_info()[2]
    # tbinfo contains the line number that the code failed on and the code from that line
    tbinfo = traceback.format_tb(tb)[0]
    # concatenate information together concerning the error into a message string
    pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
        str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
    # generate a message string for any geoprocessing tool errors
    msgs = "GP ERRORS:\n" + gp.GetMessages(2) + "\n"
    gp.AddError(msgs)

    # return gp messages for use with a script tool
    gp.AddError(pymsg)

    # print messages for use in Python/PythonWin
    print (pymsg)
    print (msgs)
    raise
#gp.MissingDataVariance_sdm(Input_Evidence_Rasters, Output_Weights_Table, crap_pprb, Cellsize)


