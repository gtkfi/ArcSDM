import arcpy, os, sys, traceback, arcgisscripting
gp = arcgisscripting.create()


def execute(self, parameters, messages):

    try:
        def unique_name(filename, part):
            base, ext = filename.split('.')
            return base + str(part) + '.' + ext        
        #input_file = 'c:\carlin_gis\combine_kbge1_class0.dta'
        input_file = parameters[0].valueAsText; #gp.GetParameterAsText(0)
        split_len = 200000
        fdin = open(input_file, 'r')
        save1 = fdin.readline()
        save2 = fdin.readline()
        save3 = fdin.readline()
        num_data = int(fdin.readline()[:-1])
        if num_data < split_len:
            raise Exception('not enough data to partition')
        else:
            num_whole_parts = num_data // split_len
            len_last_part = num_data % split_len
            len_parts = [split_len for i in range(num_whole_parts)] + [len_last_part]
            for part_n, part_len in enumerate(len_parts):
                fdout = open(unique_name(input_file, part_n+1), 'w')
                print 'processing partition %s...'%(fdout.name)
                gp.addmessage('processing partition %s...'%(fdout.name))
                fdout.write(save1)
                fdout.write(save2)
                fdout.write(save3)
                fdout.write(str(part_len) + '\n')
                for n, line in enumerate(fdin):
                    fdout.write(line)
                    if n > 1 and n % split_len == (split_len-1): break
                fdout.close()
        fdin.close()
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
