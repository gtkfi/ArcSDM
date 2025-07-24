'''

Combine nn output files
ArcSDM 5 
Converted by Tero Ronkko, GTK 2017

'''
import arcpy
import sys
import traceback


def execute(self, parameters, messages):
    try:
    ##    input_files = ['C:\Gary_stuff\Class_Partition\Class_Problem\Partition1.pnn',
    ##                   'C:\Gary_Stuff\Class_Partition\Class_Problem\Partion2.pnn']
        input_files = parameters[0].valueAsText.split(';')
    ##    output_file = 'C:\Gary_Stuff\Class_Partition\Class_Problem\NNOutput.pnn'
        output_file = parameters[1].valueAsText
        fdout = open(output_file, 'w')
        row_id = 1
        for input_file in input_files:
            fdin = open(input_file, 'r')
            print(f"processing partition {fdin.name}...")
            arcpy.AddMessage(f"processing partition {fdin.name}...")
            for line in fdin:
                items = line.split(',')
                items[0] = str(row_id)
                row_id += 1
                outline = ','.join(items)
                fdout.write(outline)
            fdin.close()
        fdout.close()     
    except Exception:
        e = sys.exc_info()
        tb = e[2]
        # tbinfo contains the line number that the code failed on and the code from that line
        tb_info = traceback.format_tb(tb)[0]
        error_message = f"PYTHON ERRORS:\nTraceback Info:\n{tb_info}\nError Info:\n{e[0]}: {e[1]}\n"
        messages.AddError(error_message)

        # print messages for use in Python/PythonWin
        print(error_message)
        print(f"GP ERRORS:\n{arcpy.GetMessages(2)}\n")
        
