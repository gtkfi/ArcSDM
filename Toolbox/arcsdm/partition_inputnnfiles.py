import arcpy
import sys
import traceback


def execute(self, parameters, messages):

    try:
        def unique_name(filename, part):
            base, ext = filename.split('.')
            return base + str(part) + '.' + ext        
        #input_file = 'c:\carlin_gis\combine_kbge1_class0.dta'
        input_file = parameters[0].valueAsText
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
                fdout = open(unique_name(input_file, part_n + 1), 'w')
                print(f"processing partition {fdout.name}...")
                arcpy.AddMessage(f"processing partition {fdout.name}...")
                fdout.write(save1)
                fdout.write(save2)
                fdout.write(save3)
                fdout.write(str(part_len) + '\n')
                for n, line in enumerate(fdin):
                    fdout.write(line)
                    if n > 1 and n % split_len == (split_len - 1):
                        break
                fdout.close()
        fdin.close()
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
