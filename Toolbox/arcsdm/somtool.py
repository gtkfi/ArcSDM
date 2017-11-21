# THis is somtest PY
#
# Tero Ronkko 2016-2017 MPM project 
#
# EXPERIMENTAL Minisom implementation for raster classification
# Input raster needs to be composite bands raster
#

import arcpy;
import numpy;
from arcsdm.minisom import MiniSom    
from datetime import datetime
import itertools;
import random;
import time;
import math;
from pylab import text,show,cm,axis,figure,subplot,imshow,zeros


def myprint (str):
    arcpy.AddMessage(str);
    print (str);


def distance(v1, v2):
    #tulos = 0;
    #myprint (" v1" + str(v1))
    #myprint (" v2" + str(v2))
    #for a1,a2 in itertools.izip(v1,v2):    
    #    tulos += abs(a1-a2);
    #return tulos / len(v1);
    return numpy.linalg.norm(v1-v2)
    
def somtrain(somsize, depth, data, iterations):
    #Reshape data as chain of data:
    datachain = data.reshape(len(data)*len(data[1]),-1);    
    myprint (" Training %s x %s x %s with source grid size %s x %s..."%(somsize, somsize, len(datachain[0]), len(data),len(data[0]) ));
    myprint (" Total of %s vectors will be iterated %s time(s)" % ( len(datachain), iterations));
    
    iterations = len(datachain) * iterations;
    starttime = time.time();
    som = MiniSom(somsize, somsize, len(datachain[0]), sigma=0.3, learning_rate=0.5) # initialization of SOM
    
    #vaihdetaan raster_array [x][y][n] tyyppiseksi on (n x y)
    #poistetaan taulukosta kokonaan ulottuvuus x
    #iterations = 100;
    #som.train_batch(datachain, iterations);
    som.train_random(datachain, iterations);
    
    endtime = time.time()
    elapsed = endtime - starttime;
    myprint ("     ... complete (took %s seconds) "%(str(math.trunc(elapsed))));
    return som;

    
def execute(self, parameters, messages):    
    myprint ("\nstarting somtest.py");
    myprint ("")
    myprint ("="*40);
    #myRaster = arcpy.Raster("composite")
    rastername = parameters[0].valueAsText;
    #myprint ("Input raster: %s"% (rastername))
    rasteri = arcpy.Raster(rastername)
    
    arcpy.env.overwriteOutput = True
    somsize  = parameters[1].value; # Maybe get this as parameter
    iterations = parameters[2].value; # Amount of iterations
    dsigma = parameters[3].value; # Amount of iterations
    dlearningrate = parameters[4].value; # Amount of iterations

    output_rastername = parameters[5].valueAsText
    
    

    raster_array = arcpy.RasterToNumPyArray (rasteri, nodata_to_value=-1);
    myprint("%-30s %s" % ("Raster name: ", rasteri));
    desc = arcpy.Describe(rastername)
    myprint("%-30s %s" % ("Raster bandcount: ", desc.bandCount));
    
    if (desc.bandCount < 2):
        arcpy.AddError("Input raster is not multiband raster!");
        return;
        #raise arcpy.ExecuteError ("Wrong input type");
    
    myprint ("%-30s %s*%s*%s" % ("Initial SOM size & depth: ", str(somsize), str(somsize), str(len(raster_array))) ) ;
    myprint("%-30s %s" % ("Training iterations: ", iterations));
    myprint("%-30s %s" % ("Learning rate: ", dlearningrate));
    myprint("%-30s %s" % ("Sigma: ", dsigma));
    myprint("%-30s %s" % ("Output rastername: ", output_rastername));
    myprint ("");

    #init som 
    
    som = MiniSom(somsize, somsize, len(raster_array), sigma=dsigma, learning_rate=dlearningrate) # initialization of SOM

    myprint ("\n\nStarting training of the SOM network...")
    starttime = time.time();

    #vaihdetaan raster_array [x][y][n] tyyppiseksi on (n x y)
    raster_array2 = numpy.swapaxes(raster_array, 0,2)

    #poistetaan taulukosta kokonaan ulottuvuus x
    #raster_datachain = raster_array2.reshape(len(raster_array2)*len(raster_array2[1]),-1)
    #myprint ("raster_datachain: " + str(raster_datachain.shape));
    #som.train_random(raster_datachain, iterations);

    som = somtrain(somsize,somsize,raster_array2,iterations);

    #somreduce = 0; # We shall do four rounds of halving the som size.
    #for round in range(0, somreduce):
    #    somsize = somsize / 2;
    #    myprint ("Iteration %s, somsize now %s" % ( str(round +1 ) , somsize) );
    #    som = somtrain(somsize,somsize,som.weights);
        
    #figure(1)
    #text(100, 200, "test", color=cm.Dark2(10 / 4.), fontdict={'weight': 'bold', 'size': 11})
    #show();
        

    #endtime = time.time()
    #elapsed = endtime - starttime;
    #myprint ("   ... complete (took %s seconds) "%(str(math.trunc(elapsed))));




    #som2 = somtrain(50,50,som.weights);
    #myprint ("Som2.weights: " + str(som2.weights.shape));
    #som3 = somtrain(30,30,som2.weights);
    #som4 = somtrain(15,15,som3.weights);
    #somfinal = somtrain (5,5, som4.weights);

    #som = somtrain(50,50,som.weights);

    #myprint ("Som2.weights: " + str(som2.weights.shape));

    #som = somtrain(30,30,som.weights);
    #som = somtrain(15,15,som.weights);

    somfinal = som



    #myprint (str(som.weights.shape));

    #uusi = numpy.zeros(( len(raster_datachain), len(raster_datachain[0]),1))
    #uusi = numpy.zeros(( len(raster_array2), len(raster_array2[0]),len(raster_array)))

    uusi = numpy.zeros(( len(raster_array2), len(raster_array2[0]),1))

    #iteroidaan

    #myprint("X:" + str(len(raster_array2)));
    #myprint("Y: " + str(len(raster_array2[0])));
    datachain = raster_array2.reshape(len(raster_array2)*len(raster_array2[1]),-1);   
    
    #Disabled at the moment
    #myprint ("\n\nCalculating distance map...");
    #starttime = time.time();
    #qemap = 0;#somfinal.quantization_error(datachain)
    #endtime = time.time()
    #elapsed = endtime - starttime;            
    #myprint ("  Result: %s ... took %s seconds\n "%(qemap, (math.trunc(elapsed))));            
                
    #myprint (str(qemap.shape));

    myprint ("\n\nFinding BMU from input rasterstack and calculating distance...");
    starttime = time.time();

    for i in range(0,len(raster_array2)):
        printed = 0;
        for j in range (0, len(raster_array2[0])):        
            a1 = raster_array2[i][j];
            w = somfinal.winner( a1 );
            
            a2 = somfinal.weights[ w[0],w[1]  ]
            #uusi[i][j] = a2[0]; 
            uusi[i][j] = distance(a2,a1);
            if (i % 50 == 0 and printed == 0 and i != 0):
                printed = 1;
                #myprint ("Row:" + str(i));    
                endtime = time.time()
                elapsed = endtime - starttime;
                ri = random.randint(0,i-1);
                rj = random.randint(0,len(raster_array2[0]-1));
                myprint ("  ... %s/%s rows processed (chunk took %s seconds). \n              Have a random result while waiting: uusi[%s][%s]=%s"%(i, len(raster_array2), str(math.trunc(elapsed)), ri, rj, str(uusi[ri][rj]) ));            
                #myprint (str(a2));
                starttime = time.time();
                #if (j % 500 == 0):
                #myprint ("i:%s j:%s a1:%s a2:%s tulos:%s "%(i,j,a1,a2,str(uusi[i][j])))
            #    myprint ("tulos:%s "%(str(uusi[i][j])))
            #    endtime = time.time()
            #    elapsed = endtime - starttime;
            #    myprint ("i: %s j:%s  ... complete (took %s seconds) "%(i,j, str(math.trunc(elapsed))));            
                #myprint (str(a2));
           #     starttime = time.time();

            #if (i % 100 == 0):
                #myprint ("pok: " + str(i) + " " + str(distance(a1,a2)));

              
    #myprint ("Uusi shape: %s"%(str(numpy.shape(uusi))));
    #myprint (numpy.shape(raster_array));
    #myprint (numpy.shape(raster_array));
    #myprint (numpy.shape(raster_array));
    myprint("\n");
                
    #numero4 = uusi.reshape( len(raster_array), len(raster_array[1]), len(raster_array[0] ));




    numero4 = numpy.swapaxes(uusi, 0,2)



    numero4 = numpy.delete(numero4, numpy.s_[1:], 0);


    mx = rasteri.extent.XMin + 0 * rasteri.meanCellWidth

    my = rasteri.extent.YMin + 0 * rasteri.meanCellHeight

    #myprint ( "Size of output raster: %s x %s"%( numpy.shape(numero4),numpy.shape(uusi)));

    #myprint (uusi[100]);

    # Overwrite
    arcpy.env.overwriteOutput = True
    myRasterBlock = arcpy.NumPyArrayToRaster(numero4, arcpy.Point(mx, my),rasteri.meanCellWidth, rasteri.meanCellHeight);

    myprint ("Output rastername: %s"% (output_rastername ));
    #myRasterBlock.save("d:\\arcgis\\database.gdb\\tulos");
    myRasterBlock.save(output_rastername);

    arcpy.SetParameter(1, myRasterBlock)    ;

                

    #myprint (elapsed); 


    #myprint (som.weights);

    myprint ("Ready!");








