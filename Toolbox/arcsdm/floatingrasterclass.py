"""This version of floatingraster searchcursor converts a float raster to a FLOAT
and HEADER file which is read back with Python's array object to generate a
pseudoVAT as a Python dictionary.
    The searchcursor is a function that returns a generator function (pseudo ROWS)
which YIELDs pseudoROWs from the pseudoVAT with OID, VALUE, and COUNT fields.
The scale of the yielded float raster VALUE is 15.
    The field RASTERVALU added to a point feature class by the EXTRACT VALUES TO
POINTS tool from a floating raster will often have a scale of 8.  Thus, values from a
RASTERVALU field can only be compared with ROUND( VALUE, scale) from a float
raster.  According to ESRI documentation, floating point values saved in field with a
scale of 6 or less have FLOAT type in tables and the type DOUBLE for higher scales.
    All attribute and function names must be lower case.
"""
import os, sys, traceback, array
import arcpy

class FloatRasterVAT(object):
    """ Pseudo VAT for a float-type raster and useful methods """
    def __init__(self, float_raster):
        """ Generator yields VAT-like rows for floating rasters """
        # Process: RasterToFLOAT_conversion: FLOAT raster to FLOAT file
        # Get a output scratch file name
        ## TODO: .flt for dir, no .flt to geodatabase!
        ## Now using Scratchfolder
        #OutAsciiFile = arcpy.CreateUniqueName("tmp_rasfloat.flt", arcpy.env.scratchWorkspace)  
        #arcpy.AddMessage(" -- Debug: FLoatingRasterArray --");
        #arcpy.AddMessage("Debug:" + arcpy.env.scratchFolder);
        
        OutAsciiFile = arcpy.CreateUniqueName("tmp_rasfloat.flt", arcpy.env.scratchFolder)
        # Convert float raster to FLOAT file and ASCII header
        arcpy.RasterToFloat_conversion(float_raster, OutAsciiFile)
        
        # Create dictionary as pseudo-VAT
        # Open ASCII header file and get raster parameters
        hdrpath = os.path.splitext(OutAsciiFile)[0] + ".hdr"
        
        try:
            fdin = open(hdrpath,'r')
            ncols = int(fdin.readline().split()[1].strip())
            nrows = int(fdin.readline().split()[1].strip())
            xllcorner =  float(fdin.readline().split()[1].strip().replace(",", ".")) #NO commas!
            yllcorner = float(fdin.readline().split()[1].strip().replace(",", "."))
            cellsize = float(fdin.readline().split()[1].strip().replace(",", "."))
            self.nodata_value = float(fdin.readline().split()[1].strip().replace(",", "."))
            byteorder = fdin.readline().split()[1].strip()
        finally:
            fdin.close()
        
        # Get FLOAT file path
        fltpath = OutAsciiFile
        # Get filesize in bytes
        filesize = os.path.getsize(fltpath)
        # Get number bytes per floating point value
        bytesperfloat = filesize/ncols/nrows
        # Set array object type
        if bytesperfloat == 4: arraytype = 'f'
        else:
            raise Exception('Unknown floating raster type')
            
        # Open FLOAT file and process rows
        try:
            fdin = open(fltpath, 'rb')
            self.vat = {}
            vat = self.vat
            for i in range(nrows):
                # Get row of raster as floating point Python array
                arry = array.array(arraytype)
                try:
                    arry.fromfile(fdin,ncols)
                except:
                    arcpy.AddError("Array input error")
                # Swap bytes, if necessary
                if byteorder != 'LSBFIRST': arry.byteswap()
                # Process raster values to get occurrence frequencies of unique values
                for j in range(ncols):
                    value = arry[j]
                    if value == self.nodata_value: continue
                    if value in vat:
                        vat[value] += 1
                    else:
                        vat[value] = 1
        finally:
            fdin.close()

    def getnodata(self):
        """ Return the NoData value of the raster """
        return self.nodata_value

    # Row definition
    class row(object):
        """ Row definition """
        def __init__(self, oid, float_, count):
            self.oid = oid
            self.value = float_
            self.count = count
        def getvalue(self, name):
            """ Get value of the row attribute by name """
            return getattr(self, name)
        def __getattribute__(self, name):
            """ Allow any capitalization of row's attributes """
            return object.__getattribute__(self,name.lower())
        def __eq__(self, testValue):
            """ Check if the row value is approximately equal to the test value """
            return abs(self.value - testValue) < 1.0e-6
            
    def __getattribute__(self, attr):
        """ Allow any capitalization of class attributes """
        return object.__getattribute__(self, attr.lower())

    def __len__(self):
        """ Return the number of unique values in the raster """
        return len(self.vat)

    def __contains__(self, testValue):
        """ Test if testValue is near enough to a raster value """
        absdiffs = [abs(testValue - rasval) for rasval in list(self.vat)]
        mindiff = min(absdiffs)
        return mindiff < 1.e-6
    
    def index(self, testValue):
        """ Return the index in the VAT's keys of raster value nearest to testValue """
        try:
            if testValue in self:
                absdiffs = [abs(testValue - rasval) for rasval in list(self.vat)]
                mindiff = min(absdiffs)
                return absdiffs.index(mindiff)
            else: raise ValueError
        except ValueError as msg:
            raise
        
    def __getitem__(self, testValue):
        """ Return the raster value nearest to testValue """
        return list(self.vat)[self.index(testValue)]
        
    def floatrastersearchcursor(self):
        """ A searchcursor for a float-type raster """
        # Generator to yield rows via Python "for statement"
        # Row returns OID, VALUE, COUNT as if pseudoVAT.
        # Raster VALUEs increasing as OID increases
        vat = self.vat
        for oid, value in enumerate(sorted(vat.keys())):
            try:
                # vat key is float value
                count = vat[value]
            except KeyError:
                print ('error value: ',repr(value))
                count = -1
            yield self.row(oid,value,count)

def FloatRasterSearchcursor(float_raster):
    """ Searchcursor from FloatRasterVAT instance """
    float_raster = FloatRasterVAT(float_raster)
    return float_raster.floatrastersearchcursor()

def rowgen(searchcursor_rows):
    """ Convert arcpy searchcursor to a generator function """
    rows = searchcursor_rows
    row = next(rows)        
    while row:
        yield row
        row = next(rows)
        
if __name__ == '__main__':    
    # Check out any necessary licenses
    arcpy.CheckOutExtension("spatial")

    arcpy.env.overwriteOutput = True

    try:
        # List all rasters starting with 'gw'
        floatrasters = arcpy.ListRasters('gw*')
        # Iterate through each raster
        for i,floatraster in enumerate(rowgen(floatrasters)):
            print (floatraster)
            flt_ras = FloatRasterVAT(floatraster)
            print (flt_ras.getnodata())
            print (0.00005 in flt_ras)
            for row in flt_ras.floatrastersearchcursor():
                print (row.value, flt_ras[row.value])
            if i>1: break
            
##        Input_raster = 'gw1a_lrconf'
##        print arcpy.Describe(Input_raster).catalogPath
##        valuetype = arcpy.GetRasterProperties (Input_raster, 'VALUETYPE')
##        valuetypes = {1:'Integer', 2:'Float'}
##        if valuetype != 2:
##            arcpy.AddError('Not a float-type raster')
##            raise
##        flt_ras = FloatRasterVAT(Input_raster)
##        print len(flt_ras)
##        tblval = 0.96671057
##        print tblval in flt_ras
##        if tblval in flt_ras:
##            try:
##                print tblval, flt_ras[tblval]
##            except ValueError as msg:
##                print msg
##            try:
##                print flt_ras.index(tblval)
##            except ValueError as msg:
##                print msg
##        rows = flt_ras.FloatrasterSearchCursor()
##        print('OID   VALUE   COUNT')
##        for row in rows:
##            print '%s %s %s' %(row.oid,round(row.value,8),row.getvalue('count')), row == tblval
##            #arcpy.AddMessage( '%s %s %s' %(row.oid,row.value,row.getvalue('count')))
##            if row.oid > 50:
##                arcpy.AddWarning('Greater than 50 raster values.')

    except:
        # get the traceback object
        tb = sys.exc_info()[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tbinfo = traceback.format_tb(tb)[0]
        # concatenate information together concerning the error into a message string
        pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + "\nError Info:\n    " + \
            str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"
        # generate a message string for any geoprocessing tool errors
        msgs = "GP ERRORS:\n" + arcpy.GetMessages(2) + "\n"
        arcpy.AddError(msgs)

        # return arcpy messages for use with a script tool
        arcpy.AddError(pymsg)

        # print messages for use in Python/PythonWin
        print (pymsg)
        print (msgs)
        raise
