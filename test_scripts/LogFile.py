import subprocess

class LogFile:


    def __init__(self,file_tool):

        #Prepare log file
        self.log_fname = "Log_" + file_tool.start_dt_string + '.txt'
        self.local_path = file_tool.local_path
        self.full_path_name = self.local_path + '/' + self.log_fname
        self.notes = file_tool.notes

        self.close_event = file_tool.close_event

        fLog = open(self.full_path_name,'w+')
        fLog.write("Run notes: " + self.notes + "\n")
        fLog.write("{}\t{}\t{}\t{}\t{}\n".format("DateTime","x1","y1","x2","y2"))
        fLog.close()





    def writeToLog(self,write_string):
        fLog = open(self.full_path_name,'a')
        fLog.write(write_string)
        fLog.close()
