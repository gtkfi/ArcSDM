"""

SDM Values / ArcSDM 6 ToolBox for ArcGIS Pro

Conversion and tool development for ArcGIS Pro by Geological Survey of Finland (GTK), 2024.

This version of floatingraster searchcursor converts a float raster to a FLOAT
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
import arcpy
import array
import os

class FloatRasterVAT(object):
    def __init__(self, float_raster, *args):
        """ Generator yields VAT-like rows for floating-point rasters """
        OutAsciiFile = arcpy.CreateUniqueName("tmp_rasfloat.flt", arcpy.env.scratchFolder)
        # Convert float raster to FLOAT file and ASCII header
        arcpy.conversion.RasterToFloat(float_raster, OutAsciiFile)

        # Open ASCII header file and get raster parameters
        hdrpath = os.path.splitext(OutAsciiFile)[0] + ".hdr"
        try:
            fdin = open(hdrpath, 'r')
            ncols = int(fdin.readline().split()[1].strip())
            nrows = int(fdin.readline().split()[1].strip())
            xllcorner =  float(fdin.readline().split()[1].strip().replace(",", "."))
            yllcorner = float(fdin.readline().split()[1].strip().replace(",", "."))
            cellsize = float(fdin.readline().split()[1].strip())
            self.NODATA_value = int(fdin.readline().split()[1].strip())
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
        if bytesperfloat == 4:
            arraytype = 'f'
        else:
            raise Exception('Unknown floating raster type')
            
        # Open FLOAT file and process rows
        try:
            fdin = open(fltpath, 'rb')
            # Create dictionary as pseudo-VAT
            self.vat = {}
            vat = self.vat
            for i in range(nrows):
                # Get row of float raster file as a floating-point Python array
                arry = array.array(arraytype)
                try:
                    arry.fromfile(fdin, ncols)
                except:
                    arcpy.AddError("Array input error")
                # Swap bytes, if necessary
                if byteorder != 'LSBFIRST':
                    arry.byteswap()
                # Process raster values to get occurence frequencies of unique values
                for j in range(ncols):
                    value = arry[j]
                    if value == self.NODATA_value:
                        continue
                    if value in vat:
                        vat[value] += 1
                    else:
                        vat[value] = 1
        finally:
            fdin.close()
        
        # Delete the created files
        arcpy.management.Delete(OutAsciiFile)
        arcpy.management.Delete(hdrpath)

    def getNODATA(self):
        return self.NODATA_value

    # Row definition
    class row(object):
        """ row definition """
        def __init__(self, oid, float_, count):
            self.oid = oid
            self.value = float_
            self.count = count
        def getvalue(self, name):
            # TODO: getvalue doesn't do what it should?
            return getattr(self, name)
        def __getattr__(self, name):
            """ Allow any capitalization of row's attributes """
            return getattr(self, name.lower())
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
            else:
                raise ValueError
        except ValueError:
            raise
        
    def __getitem__(self, testValue):
        """ Return raster value nearest testValue  """
        return self.vat.keys()[self.index(testValue)]
        
    def FloatRasterSearchcursor(self):
        """ Return a generator function that produces searchcursor rows from VAT """
        # Generator to yield rows via Python "for" statement
        # Row returns OID, VALUE, COUNT as if pseudoVAT.
        # Raster VALUEs increasing as OID increases
        vat = self.vat
        for oid, value in enumerate(sorted(vat.keys())):
            try:
                # vat key is float value
                count = vat[value]
            except KeyError:
                print('error value: ', repr(value))
                count = -1
            yield self.row(oid, value, count)


def FloatRasterSearchcursor(float_raster, *args):
    """ Return a searchcursor from FloatRasterVAT instance """
    float_raster = FloatRasterVAT(float_raster, args)
    return float_raster.FloatRasterSearchcursor()
