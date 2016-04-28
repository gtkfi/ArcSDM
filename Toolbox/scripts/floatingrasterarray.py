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
    VAT does not include the NODATA value.
    Use getNODATA() to get NODATA value.
"""
import os, sys, traceback, array

class FloatRasterVAT(object):
    def __init__(self, gp, float_raster, *args):
        """ Generator yields VAT-like rows for floating-point rasters """
        # Process: RasterToFLOAT_conversion: FLOAT raster to FLOAT file
        # Get a output scratch file name
        OutAsciiFile = gp.createuniquename("tmp_rasfloat.flt", gp.scratchworkspace)
        #Convert float raster to FLOAT file and ASCII header
        gp.RasterToFLOAT_conversion(float_raster, OutAsciiFile)
        # Create dictionary as pseudo-VAT
        # Open ASCII header file and get raster parameters
        print (OutAsciiFile)
        hdrpath = os.path.splitext(OutAsciiFile)[0] + ".hdr"
        print (hdrpath)
        try:
            fdin = open(hdrpath,'r')
            self.ncols = int(fdin.readline().split()[1].strip())
            self.nrows = int(fdin.readline().split()[1].strip())
            self.xllcorner =  float(fdin.readline().split()[1].strip())
            self.yllcorner = float(fdin.readline().split()[1].strip())
            self.cellsize = float(fdin.readline().split()[1].strip())
            self.NODATA_value = int(fdin.readline().split()[1].strip())
            self.byteorder = fdin.readline().split()[1].strip()
        finally:
            fdin.close()
        #Get FLOAT file path
        fltpath = OutAsciiFile
        #Get filesize in bytes
        filesize = os.path.getsize(fltpath)
        #Get number bytes per floating point value
        bytesperfloat = filesize/self.ncols/self.nrows
        #Set array object type
        if bytesperfloat == 4: arraytype = 'f'
        else:
            raise Exception ('Unknown floating raster type')
            
        #Open FLOAT file and process rows
        try:
            fdin = open(fltpath, 'rb')
            self.vat = {}
            vat = self.vat
            for i in range(self.nrows):
                #Get row of float raster file as a floating-point Python array
                arry = array.array(arraytype)
                arry.fromfile(fdin, self.ncols)
                #Swap bytes, if necessary
                if self.byteorder != 'LSBFIRST': arry.byteswap()
                #Process raster values to get occurence frequencies of unique values
                for j in range(self.ncols):
                    value = arry[j]
                    if value == self.NODATA_value: continue
                    if value in vat:
                        vat[value] += 1
                    else:
                        vat[value] = 1
        finally:
            fdin.close()
        #print 'Unique values count in floating raster = %s'%len(vat)
        #print len(vat),min(vat.keys()),max(vat.keys())
        #print vat

    def getNODATA(self):
        return self.NODATA_value

    #Row definition
    class row(object):
        """ row definition """
        def __init__(self, oid, float_, count):
            self.oid = oid
            self.value = float_
            self.count = count
        def getvalue(self, name):
            return getattr(self, name)
        def __getattr__(self, name):
            """ Allow any capitalization of row's attributes """
            return getattr(self,name.lower())
        def __eq__(self, testValue):
            return abs(self.value - testValue) < 1.0e-6
            

    def __len__(self):
        """ Return row count of VAT """
        return len(self.vat)

    def __contains__(self, testValue):
        """ Return if testValue is near a raster value """
        absdiffs = [abs(testValue - rasval) for rasval in self.vat.keys()]
        mindiff = min(absdiffs)
        self._index = absdiffs.index(mindiff)
        return mindiff < 1.e-6
    
    def index(self, testValue):
        """ Return index in VAT keys of raster value nearest testValue  """
        try:
            if testValue in self:
                return self._index
            else: raise ValueError
        except ValueError (msg):
            raise
        
    def __getitem__(self, testValue):
        """ Return raster value nearest testValue  """
        return self.vat.keys()[self.index(testValue)]
        
    def FloatRasterSearchcursor(self):
        """ Return a generator function that produces searchcursor rows from VAT """
        #Generator to yield rows via Python "for" statement
        #Row returns OID, VALUE, COUNT as if pseudoVAT.
        #Raster VALUEs increasing as OID increases
        vat = self.vat
        for oid, value in enumerate(sorted(vat.keys())):
            try:
                #vat key is float value
                count = vat[value]
            except KeyError:
                print ('error value: ',repr(value))
                count = -1
            yield self.row(oid,value,count)

def FloatRasterSearchcursor(gp, float_raster, *args):
    """ Return a searchcursor from FloatRasterVAT instance """
    float_raster = FloatRasterVAT(gp, float_raster, args)
    return float_raster.FloatRasterSearchcursor()

# Local function and variables...

def rowgen(rows):
    """ Convert a gp searchcursor to a generator function """
    row = rows.next()        
    while row:
        yield row
        row = rows.next()

if __name__ == '__main__':
    
    import arcgisscripting
    gp = arcgisscripting.create()
    # Check out any necessary licenses
    gp.CheckOutExtension("spatial")

    gp.OverwriteOutput = 1

    try:
        gp.workspace = "C:/Saibal_stuff/Saibal's_data"
        gp.scratchworkspace = "C:/TEMP"
        Input_raster = 'w_pprb6'
        print (gp.describe(Input_raster).catalogpath)
        valuetype = gp.GetRasterProperties (Input_raster, 'VALUETYPE')
        valuetypes = {1:'Integer', 2:'Float'}
        if valuetype != 2:
            gp.adderror('Not a float-type raster')
            raise
        flt_ras = FloatRasterVAT(gp, Input_raster)
        print (len(flt_ras))
        tblval = flt_ras.getNODATA()
        print (tblval in flt_ras)
        try:
            print (tblval, flt_ras[tblval])
        except ValueError as msg:
            print (tblval, 'value not found')
        try:
            print (flt_ras.index(tblval))
        except ValueError as msg:
            print (tblval, 'index not found')
##        rows = flt_ras.FloatRasterSearchcursor()
##        rows = FloatRasterSearchcursor(gp, os.path.join(gp.workspace, Input_raster))
##        print ('OID   VALUE   COUNT   T/F')
##        for row in rows:
##            print '%d %.16e %d' %(row.OID,row.value,row.getvalue('Count')), row == tblval
##            #gp.addmessage( '%s %s %s' %(row.OID,row.Value,row.getvalue('Count')))
##            if row.oid > 50:
##                gp.AddWarning('Greater than 50 raster values.')
##                break

    except:
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
