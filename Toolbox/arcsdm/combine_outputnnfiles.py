'''

Combine nn output files
ArcSDM 5 
Converted by Tero Ronkko, GTK 2017



'''
import arcpy,os, sys, traceback, arcgisscripting
gp = arcgisscripting.create()


def execute(self, parameters, messages):

    try:
    ##    input_files = ['C:\Gary_stuff\Class_Partition\Class_Problem\Partition1.pnn',
    ##                   'C:\Gary_Stuff\Class_Partition\Class_Problem\Partion2.pnn']
        input_files = parameters[0].valueAsText.split(';'); #gp.GetParameterAsText(0).split(';')
    ##    output_file = 'C:\Gary_Stuff\Class_Partition\Class_Problem\NNOutput.pnn'
        output_file = parameters[1].valueAsText; #gp.GetParameterAsText(1)
        fdout = open(output_file,'w')
        row_id = 1
        for input_file in input_files:
            fdin = open(input_file,'r')
            print 'processing partition %s...'%(fdin.name)
            gp.addmessage( 'processing partition %s...'%fdin.name)
            for line in fdin:
                items = line.split(',')
                items[0] = str(row_id)
                row_id += 1
                outline = ','.join(items)
                fdout.write(outline)
            fdin.close()
        fdout.close()
            
    except Exception, msg:
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
        print pymsg
        print msgs
        raise
        
